"""
Credit Lending Strategy - Can a fintech-style model approve more good borrowers?
===============================================================================
Challenges the bank-style scorecard from credit-risk-scorecard (project 1) with an
XGBoost model on the same borrowers, then turns both models into lending decisions:
  - same cleaning and the same 80/20 split as project 1, so the comparison is fair
  - AUC / Gini / KS and 10-decile calibration for both models
  - why class weighting (scale_pos_weight) is not used: it breaks the PDs
  - approval cut-off strategy: approval rate vs default rate vs profit
  - who gets approved differently: the "swap set" between the two models
  - monotone constraints so every risk factor moves in a direction a lender can explain
  - SHAP contributions (built into XGBoost) for drivers and reason codes

Run: python strategy.py
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from baseline.scorecard import (FLAG_COLS, LABELS, SEED, TARGET, apply_caps, clean,
                                pd_to_score)

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "credit_data.csv"
BASELINE_MODEL = ROOT / "baseline" / "outputs" / "scorecard_model.pkl"
OUT_DIR = ROOT / "outputs"

LABELS = {**LABELS, "DebtRatio": "Debt ratio"}

# ---------------------------------------------------------------------------
# Business assumptions (illustrative, clearly labelled)
# Product: a credit card line, same as project 1's expected-loss assumptions.
# ---------------------------------------------------------------------------
CREDIT_LIMIT = 10_000   # assumed limit per approved borrower ($)
CCF = 0.50              # EAD = drawn balance + 50% of the unused limit
LGD = 0.80              # share of exposure lost when a borrower defaults
APR = 0.20              # interest rate charged on the drawn balance
COST_OF_FUNDS = 0.06    # funding + servicing cost per year
YEARS = 2               # matches the two-year default horizon
PER = 100_000           # results are scaled to "per 100,000 applicants"

# XGBoost settings: sensible defaults, no hyperparameter search
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.05, max_depth=4,
                  subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                  eval_metric="logloss", n_jobs=4)

# Monotone constraints: +1 = can only raise risk, -1 = can only lower it, 0 = free.
# They keep every factor explainable (e.g. older applicants are never scored as riskier,
# matching the Reg B rule on age) and cost about 0.002 AUC.
MONOTONE = {
    "RevolvingUtilizationOfUnsecuredLines": 1,
    "NumberOfTime30-59DaysPastDueNotWorse": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 1,
    "NumberOfTimes90DaysLate": 1,
    "DebtRatio": 1,
    "age": -1,
    "MonthlyIncome": -1,
}
RISK_APPETITES = [0.015, 0.02, 0.025, 0.03]   # target default rates among approved
SWAP_RATES = [70, 80, 90]                      # approval rates for the swap-set check

BLUE, ORANGE = "#2a78d6", "#eb6834"      # scorecard, XGBoost
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def line(char="=", n=78):
    print(char * n)


def ks_stat(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return np.max(tpr - fpr)


def decile_table(y, p):
    t = (pd.DataFrame({"pd": p, "y": y, "Decile": pd.qcut(p, 10, labels=range(1, 11))})
         .groupby("Decile", observed=True)
         .agg(Predicted=("pd", "mean"), Actual=("y", "mean")))
    t["Ratio"] = t["Predicted"] / t["Actual"]
    return t


def strategy_curve(p, y, profit_good, loss_bad):
    """Approve the lowest-PD applicants first. For every approval rate from 1% to 100%,
    return the default rate among approved borrowers and the realised profit."""
    order = np.argsort(p)
    y_s, g_s, l_s = y[order], profit_good[order], loss_bad[order]
    n = len(p)
    cum_bad = np.cumsum(y_s)
    cum_profit = np.cumsum(np.where(y_s == 1, -l_s, g_s))
    rows = []
    for rate in range(1, 101):
        k = int(round(n * rate / 100))
        rows.append({"Approval_rate": rate / 100, "Approved": k,
                     "Default_rate": cum_bad[k - 1] / k,
                     "Profit": cum_profit[k - 1],
                     "PD_cutoff": p[order][k - 1]})
    return pd.DataFrame(rows)


def max_approvals(p, y, profit_good, loss_bad, target):
    """Largest number of lowest-PD applicants whose default rate stays at or below
    the target, computed borrower by borrower. Returns (approved, profit)."""
    order = np.argsort(p)
    y_s = y[order]
    rate = np.cumsum(y_s) / np.arange(1, len(p) + 1)
    k = int(np.nonzero(rate <= target)[0].max()) + 1
    profit = np.where(y_s[:k] == 1, -loss_bad[order][:k], profit_good[order][:k]).sum()
    return k, profit


def contributions(model, X):
    """SHAP values: XGBoost computes them natively (last column = base value)."""
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)[:, :-1]


def fmt_value(feat, val):
    if pd.isna(val):
        return "missing"
    if feat == "RevolvingUtilizationOfUnsecuredLines":
        return f"{val:.0%} of limit used"
    if feat == "MonthlyIncome":
        return f"${val:,.0f}/month"
    if feat == "DebtRatio":
        return f"{val:.2f}"
    return f"{val:.0f}"


def risk_factors(contrib_row, x_row, n=3):
    reasons = []
    for i in np.argsort(contrib_row)[::-1]:
        if contrib_row[i] <= 0 or len(reasons) == n:
            break
        feat = x_row.index[i]
        if feat in FLAG_COLS:
            if x_row.iloc[i] == 1:
                reasons.append(LABELS[feat])
        else:
            reasons.append(f"{LABELS[feat]} ({fmt_value(feat, x_row.iloc[i])})")
    return reasons


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, PercentFormatter

    OUT_DIR.mkdir(exist_ok=True)

    # 1. Same data, same cleaning, same split as project 1 ---------------------
    df = clean(pd.read_csv(DATA_FILE))
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED)
    y_tr, y_te = y_train.values, y_test.values

    line()
    print("CREDIT LENDING STRATEGY - SCORECARD vs XGBOOST")
    line()
    print(f"Train: {len(X_train):,} borrowers | Test: {len(X_test):,} | "
          f"Default rate: {y.mean():.2%} (same split as project 1)")

    # 2. Baseline: project 1's saved scorecard, loaded as-is --------------------
    with open(BASELINE_MODEL, "rb") as f:
        base = pickle.load(f)
    X_test_base = apply_caps(X_test, base["caps"])[base["features"]]
    pd_lr = base["pipeline"].predict_proba(X_test_base)[:, 1]

    # 3. Challenger: XGBoost on the same features plus debt ratio ---------------
    #    Trees need no scaling or capping and handle missing values directly.
    xgb_features = base["features"] + ["DebtRatio"]
    constraints = tuple(MONOTONE.get(f, 0) for f in xgb_features)
    model = xgb.XGBClassifier(**XGB_PARAMS, monotone_constraints=constraints)
    model.fit(X_train[xgb_features], y_tr)
    pd_xgb = model.predict_proba(X_test[xgb_features])[:, 1]

    # Comparison only: the same model without monotone constraints
    free = xgb.XGBClassifier(**XGB_PARAMS)
    free.fit(X_train[xgb_features], y_tr)
    pd_free = free.predict_proba(X_test[xgb_features])[:, 1]

    # Comparison only: the "one line fixes imbalance" setting
    spw = (y_tr == 0).sum() / (y_tr == 1).sum()
    weighted = xgb.XGBClassifier(**XGB_PARAMS, monotone_constraints=constraints,
                                 scale_pos_weight=spw)
    weighted.fit(X_train[xgb_features], y_tr)
    pd_w = weighted.predict_proba(X_test[xgb_features])[:, 1]

    # 4. Head-to-head ------------------------------------------------------------
    models = {"Scorecard": pd_lr, "XGBoost": pd_xgb, "XGBoost weighted": pd_w}
    head = {"Scorecard": pd_lr, "XGBoost": pd_xgb, "XGBoost unconstrained": pd_free,
            "XGBoost weighted": pd_w}
    print()
    line()
    print("HEAD-TO-HEAD (30,000 hold-out borrowers)")
    line()
    print(f"{'Model':<24}{'AUC':>8}{'Gini':>8}{'KS':>8}{'Avg PD':>10}{'Actual':>10}")
    line("-")
    metrics = {}
    for name, p in head.items():
        auc = roc_auc_score(y_te, p)
        metrics[name] = {"AUC": auc, "Gini": 2 * auc - 1, "KS": ks_stat(y_te, p),
                         "Avg_PD": p.mean()}
        print(f"{name:<24}{auc:>8.3f}{2 * auc - 1:>8.3f}{metrics[name]['KS']:>8.3f}"
              f"{p.mean():>10.2%}{y_te.mean():>10.2%}")
    line("-")
    lift = metrics["XGBoost"]["AUC"] - metrics["Scorecard"]["AUC"]
    print(f"AUC lift, XGBoost over scorecard: {lift:+.3f} "
          f"({lift / metrics['Scorecard']['AUC']:+.1%} relative)")
    print(f"Monotone constraints cost {metrics['XGBoost unconstrained']['AUC'] - metrics['XGBoost']['AUC']:.3f} AUC "
          f"(unconstrained {metrics['XGBoost unconstrained']['AUC']:.3f})")
    print(f"scale_pos_weight = {spw:.1f}: AUC {metrics['XGBoost weighted']['AUC']:.3f}, "
          f"but average PD {pd_w.mean():.1%} vs {y_te.mean():.1%} actual "
          f"({pd_w.mean() / y_te.mean():.1f}x too high)")

    dec = {name: decile_table(y_te, p) for name, p in models.items()}
    print()
    line()
    print("CALIBRATION BY DECILE - predicted PD / actual default rate (1.00 = perfect)")
    line()
    print(f"{'Decile':<8}{'Actual rate':>13}" + "".join(f"{n:>19}" for n in models))
    line("-")
    for d in range(1, 11):
        print(f"{d:<8}{dec['XGBoost'].loc[d, 'Actual']:>13.2%}"
              + "".join(f"{dec[n].loc[d, 'Ratio']:>19.2f}" for n in models))
    line("-")
    print("Actual rate shown for XGBoost's deciles; each model's ratio uses its own deciles.")

    # 5. Lending strategy: approval rate vs default rate vs profit ---------------
    util = X_test["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1).values
    drawn = util * CREDIT_LIMIT
    ead = drawn + CCF * (CREDIT_LIMIT - drawn)
    profit_good = drawn * (APR - COST_OF_FUNDS) * YEARS     # interest margin earned
    loss_bad = LGD * ead                                     # loss if they default
    scale = PER / len(y_te)

    curves = {n: strategy_curve(models[n], y_te, profit_good, loss_bad)
              for n in ["Scorecard", "XGBoost"]}

    print()
    line()
    print("LENDING STRATEGY - approve the lowest-risk applicants first (per 100,000 applicants)")
    line()
    print(f"Assumptions: ${CREDIT_LIMIT:,} limit, APR {APR:.0%}, cost of funds {COST_OF_FUNDS:.0%}, "
          f"LGD {LGD:.0%}, CCF {CCF:.0%}, {YEARS}-year horizon")
    print(f"{'Approve':<9}{'Default rate':>26}{'Profit ($ millions)':>34}")
    print(f"{'':<9}{'Scorecard':>13}{'XGBoost':>13}{'Scorecard':>17}{'XGBoost':>17}")
    line("-")
    for rate in [50, 60, 70, 80, 85, 90, 95, 100]:
        a, b = curves["Scorecard"].iloc[rate - 1], curves["XGBoost"].iloc[rate - 1]
        print(f"{rate:>3}%{'':<5}{a['Default_rate']:>13.2%}{b['Default_rate']:>13.2%}"
              f"{a['Profit'] * scale / 1e6:>17.2f}{b['Profit'] * scale / 1e6:>17.2f}")
    line("-")

    best = {n: c.loc[c["Profit"].idxmax()] for n, c in curves.items()}
    print("\nProfit-maximising cut-off:")
    for n, r in best.items():
        print(f"  {n:<10} approve {r['Approval_rate']:.0%} (PD cut-off {r['PD_cutoff']:.1%}) | "
              f"default rate {r['Default_rate']:.2%} | profit ${r['Profit'] * scale / 1e6:.2f}M")
    gain = (best["XGBoost"]["Profit"] - best["Scorecard"]["Profit"]) * scale
    print(f"  Extra profit with XGBoost: ${gain / 1e6:.2f}M per 100,000 applicants "
          f"({gain / (best['Scorecard']['Profit'] * scale):+.1%})")

    # Risk appetite: at a given target default rate, how many can each model approve?
    print()
    line()
    print("RISK APPETITE - most applicants each model can approve at a target default rate")
    line()
    print(f"{'Target':<9}{'Approval rate':>24}{'Extra approvals':>18}{'Profit ($ millions)':>27}")
    print(f"{'default':<9}{'Scorecard':>12}{'XGBoost':>12}{'per 100,000':>18}{'Scorecard':>14}{'XGBoost':>13}")
    line("-")
    appetite = []
    n_te = len(y_te)
    for target in RISK_APPETITES:
        k_lr, pr_lr = max_approvals(pd_lr, y_te, profit_good, loss_bad, target)
        k_x, pr_x = max_approvals(pd_xgb, y_te, profit_good, loss_bad, target)
        extra = (k_x - k_lr) * scale
        appetite.append({"Target_default_rate": target,
                         "Scorecard_approval": k_lr / n_te, "XGBoost_approval": k_x / n_te,
                         "Extra_approvals_per_100k": extra,
                         "Scorecard_profit": pr_lr * scale, "XGBoost_profit": pr_x * scale})
        print(f"{target:<9.1%}{k_lr / n_te:>12.1%}{k_x / n_te:>12.1%}{extra:>+18,.0f}"
              f"{pr_lr * scale / 1e6:>14.2f}{pr_x * scale / 1e6:>13.2f}")
    line("-")

    # 6. Who gets approved differently? Compared at the SAME approval rate, so the
    #    two "only approved by one model" groups are the same size.
    print()
    line()
    print("SWAP SET - same approval rate, applicants the two models decide differently")
    line()
    print(f"{'Approve':<9}{'Borrowers':>11}{'Default rate':>28}{'Credit used (avg)':>28}")
    print(f"{'':<9}{'in each':>11}{'XGBoost-only':>14}{'Scorecard-only':>16}{'XGBoost-only':>14}{'Scorecard-only':>16}")
    line("-")
    swaps = []
    for rate in SWAP_RATES:
        k = int(round(len(y_te) * rate / 100))
        appr_lr = np.zeros(len(y_te), bool)
        appr_x = np.zeros(len(y_te), bool)
        appr_lr[np.argsort(pd_lr)[:k]] = True
        appr_x[np.argsort(pd_xgb)[:k]] = True
        x_only, lr_only = appr_x & ~appr_lr, appr_lr & ~appr_x
        swaps.append({"Approval_rate": rate / 100, "Borrowers_each": int(x_only.sum()),
                      "XGBoost_only_default": y_te[x_only].mean(),
                      "Scorecard_only_default": y_te[lr_only].mean(),
                      "XGBoost_only_utilization": util[x_only].mean(),
                      "Scorecard_only_utilization": util[lr_only].mean()})
        sw = swaps[-1]
        print(f"{rate:>3}%{'':<5}{sw['Borrowers_each']:>11,}{sw['XGBoost_only_default']:>14.2%}"
              f"{sw['Scorecard_only_default']:>16.2%}{sw['XGBoost_only_utilization']:>14.0%}"
              f"{sw['Scorecard_only_utilization']:>16.0%}")
    line("-")
    print("XGBoost-only = approved by XGBoost but declined by the scorecard, and vice versa.")

    # Who are they? Profile of the two swap groups at the first approval rate
    k = int(round(len(y_te) * SWAP_RATES[0] / 100))
    appr_lr = np.zeros(len(y_te), bool)
    appr_x = np.zeros(len(y_te), bool)
    appr_lr[np.argsort(pd_lr)[:k]] = True
    appr_x[np.argsort(pd_xgb)[:k]] = True
    any_late = (X_test[["NumberOfTime30-59DaysPastDueNotWorse",
                        "NumberOfTime60-89DaysPastDueNotWorse",
                        "NumberOfTimes90DaysLate"]].fillna(0).sum(axis=1) > 0).values
    print(f"\nProfile at {SWAP_RATES[0]}% approval (medians; late payment = share of group):")
    print(f"{'Group':<16}{'Default rate':>14}{'Credit used':>13}{'Any late payment':>18}"
          f"{'Age':>6}{'Income':>10}")
    for label, mask in [("XGBoost-only", appr_x & ~appr_lr), ("Scorecard-only", appr_lr & ~appr_x)]:
        d = X_test[mask]
        print(f"{label:<16}{y_te[mask].mean():>14.2%}"
              f"{d['RevolvingUtilizationOfUnsecuredLines'].median():>13.0%}"
              f"{any_late[mask].mean():>18.0%}{d['age'].median():>6.0f}"
              f"{d['MonthlyIncome'].median():>10,.0f}")

    # 7. Explainability: SHAP contributions --------------------------------------
    shap_test = contributions(model, X_test[xgb_features])
    importance = (pd.DataFrame({"Feature": xgb_features,
                                "Mean_abs_SHAP": np.abs(shap_test).mean(axis=0)})
                  .sort_values("Mean_abs_SHAP", ascending=False).reset_index(drop=True))
    importance.insert(0, "Rank", range(1, len(importance) + 1))
    print()
    line()
    print("XGBOOST RISK DRIVERS (mean absolute SHAP value, log-odds)")
    line()
    for _, r in importance.iterrows():
        print(f"{r['Rank']:<4}{r['Feature']:<42}{r['Mean_abs_SHAP']:>8.3f}")

    # Reason codes for a real declined applicant from the hold-out set
    declined = np.where(pd_xgb > best["XGBoost"]["PD_cutoff"])[0]
    i = declined[np.argsort(pd_xgb[declined])[len(declined) // 2]]   # median declined
    x_row = X_test[xgb_features].iloc[i]
    print(f"\nExample declined applicant (median PD among XGBoost declines at its best cut-off): "
          f"PD {pd_xgb[i]:.1%}, score {pd_to_score(pd_xgb[i]):.0f} "
          f"(scorecard PD {pd_lr[i]:.1%})")
    for k, r in enumerate(risk_factors(shap_test[i], x_row), 1):
        print(f"   Reason {k}: {r}")

    # 8. Save -------------------------------------------------------------------
    model.save_model(OUT_DIR / "xgboost_model.json")
    out = curves["Scorecard"][["Approval_rate", "Default_rate", "Profit"]].rename(
        columns={"Default_rate": "Scorecard_default_rate", "Profit": "Scorecard_profit"})
    out["XGBoost_default_rate"] = curves["XGBoost"]["Default_rate"]
    out["XGBoost_profit"] = curves["XGBoost"]["Profit"]
    out[["Scorecard_profit", "XGBoost_profit"]] *= scale
    out.to_csv(OUT_DIR / "strategy_curve.csv", index=False)
    importance.to_csv(OUT_DIR / "xgboost_drivers.csv", index=False)
    pd.DataFrame(appetite).to_csv(OUT_DIR / "risk_appetite.csv", index=False)
    pd.DataFrame(swaps).to_csv(OUT_DIR / "swap_set.csv", index=False)

    # 9. Charts --------------------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})

    def finish(ax, fig, name, title, grid_axis="both"):
        ax.set_title(title, loc="left", color=INK)
        ax.grid(color=GRID, lw=0.6, axis=grid_axis)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(OUT_DIR / name)
        plt.close(fig)

    # Approval rate vs default rate (the lender's trade-off)
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    for n, col in [("Scorecard", BLUE), ("XGBoost", ORANGE)]:
        c = curves[n]
        ax.plot(c["Approval_rate"], c["Default_rate"], color=col, lw=2, label=n)
    ax.set(xlabel="Share of applicants approved", ylabel="Default rate among approved",
           xlim=(0.3, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.set_ylim(0, curves["Scorecard"]["Default_rate"].max() * 1.05)
    ax.legend(frameon=False, loc="upper left")
    finish(ax, fig, "approval_vs_default.png", "Approval rate vs default rate")

    # Profit by approval rate
    fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=150)
    for n, col in [("Scorecard", BLUE), ("XGBoost", ORANGE)]:
        c = curves[n]
        ax.plot(c["Approval_rate"], c["Profit"] * scale / 1e6, color=col, lw=2, label=n)
        b = best[n]
        ax.plot(b["Approval_rate"], b["Profit"] * scale / 1e6, "o", ms=8, color=col,
                markeredgecolor="white", markeredgewidth=2)
    ax.set(xlabel="Share of applicants approved", ylabel="Profit per 100,000 applicants",
           xlim=(0.3, 1))
    ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:.0f}M"))
    ax.legend(frameon=False, loc="upper left")
    finish(ax, fig, "profit_by_cutoff.png", "Profit by approval cut-off (dots = best)")

    # XGBoost risk drivers
    top = importance.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    ax.barh([LABELS[f] for f in top["Feature"]], top["Mean_abs_SHAP"], color=ORANGE,
            height=0.6)
    ax.set(xlabel="Average impact on default risk (mean |SHAP|, log-odds)")
    finish(ax, fig, "xgboost_drivers.png", "XGBoost risk drivers", grid_axis="x")

    print("\nSaved to outputs/: xgboost_model.json, strategy_curve.csv, xgboost_drivers.csv, "
          "risk_appetite.csv, swap_set.csv, approval_vs_default.png, profit_by_cutoff.png, "
          "xgboost_drivers.png")


if __name__ == "__main__":
    main()
