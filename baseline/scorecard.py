"""
Credit Risk Scorecard - Traditional Bank Style
==============================================
Interpretable logistic regression PD model with:
  - leakage-safe preprocessing (split first, fit on train only)
  - missing-value flags + median imputation + outlier capping + standardization
  - AUC / Gini / KS validation and a 10-decile calibration check
  - Risk Weight Matrix (ranked standardized coefficients)
  - PD -> 300-850 score conversion (points-to-double-odds)
  - Expected Loss = PD x LGD x EAD (illustrative LGD / EAD assumptions)
  - score_applicant(): score, PD and top-3 reason codes for a new applicant

Data: Kaggle "Give Me Some Credit" (cs-training.csv), saved as data/credit_data.csv
Run:  python scorecard.py                     (trains, validates, saves outputs/)
Use:  from scorecard import score_applicant   (scores one applicant with the saved model)
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "credit_data.csv"
OUT_DIR = ROOT / "outputs"
MODEL_FILE = OUT_DIR / "scorecard_model.pkl"
MATRIX_FILE = OUT_DIR / "risk_weight_matrix.csv"
TARGET = "SeriousDlqin2yrs"  # 1 = 90+ days past due within 2 years
SEED = 42

# Score scaling: 650 points = 20:1 good:bad odds, every 50 points doubles the odds
BASE_SCORE, BASE_ODDS, PDO = 650, 20, 50
SCORE_MIN, SCORE_MAX = 300, 850
FACTOR = PDO / np.log(2)
OFFSET = BASE_SCORE - FACTOR * np.log(BASE_ODDS)

# Expected Loss = PD x LGD x EAD
# The dataset has no balances or recoveries, so LGD and EAD use illustrative,
# industry-typical assumptions for unsecured credit cards (clearly labelled).
CREDIT_LIMIT = 10_000   # assumed limit per borrower ($)
CCF = 0.50              # credit conversion factor: share of unused limit drawn before default
LGD = 0.80              # share of exposure lost after default (unsecured cards ~75-85%)

PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
FLAG_COLS = ["PastDueCodeFlag", "IncomeMissingFlag"]

# Debt ratio raises risk on its own, but its weight turns slightly negative once the
# other debt features are in the model, and it adds no accuracy (AUC 0.860 with it,
# 0.861 without). Every feature in the model must be one we'd show an applicant as a
# reason, so it is left out rather than kept and hidden.
DROPPED = ["DebtRatio"]

# Plain-English names, used in reason codes
LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Revolving credit utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Payments 30-59 days late",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines and loans",
    "NumberOfTimes90DaysLate": "Payments 90+ days late",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Payments 60-89 days late",
    "NumberOfDependents": "Number of dependents",
    "PastDueCodeFlag": "Past-due history recorded as a special code",
    "IncomeMissingFlag": "Income not reported",
}


def line(char="=", n=76):
    print(char * n)


# ---------------------------------------------------------------------------
# Shared helpers - used both in training and when scoring a new applicant
# ---------------------------------------------------------------------------
def clean(data):
    """Rule-based cleaning: flag missing / coded values before imputing them."""
    data = data.drop(columns=[c for c in data.columns if c.startswith("Unnamed")]).copy()
    # 96 / 98 in the past-due counts are data-entry codes, not real counts. These
    # borrowers default far more often than average, so keep that signal as a flag
    # (the three columns are always coded together, so one flag covers them).
    data["PastDueCodeFlag"] = (data[PAST_DUE_COLS] >= 96).any(axis=1).astype(int)
    for col in PAST_DUE_COLS:
        data.loc[data[col] >= 96, col] = np.nan
    # Missing income is flagged so the model can tell it apart from a typical income
    data["IncomeMissingFlag"] = data["MonthlyIncome"].isna().astype(int)
    return data


def apply_caps(data, caps):
    data = data.copy()
    for c, (lo, hi) in caps.items():
        data[c] = data[c].clip(lo, hi)
    return data


def pd_to_score(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.clip(np.round(OFFSET + FACTOR * np.log((1 - p) / p)), SCORE_MIN, SCORE_MAX)


def reason_codes(pipeline, X_row, n=3):
    """Top-n features pushing this applicant's risk UP versus the average borrower.
    For a logistic model, each feature's contribution = coefficient x standardized value."""
    z = pipeline[:-1].transform(X_row)[0]                  # imputed + standardized
    contrib = pipeline.named_steps["logit"].coef_[0] * z
    reasons = []
    for i in np.argsort(contrib)[::-1]:
        if contrib[i] <= 0 or len(reasons) == n:
            break
        feat = X_row.columns[i]
        if feat in FLAG_COLS:
            if X_row.iloc[0, i] == 1:            # a flag is only a factor when it is set
                reasons.append(LABELS[feat])
        else:
            reasons.append(f"{LABELS[feat]}: {'higher' if z[i] > 0 else 'lower'} than average")
    return reasons


def score_applicant(applicant, bundle=None):
    """Score one applicant (a dict of the raw dataset fields; leave unknowns as None).
    Returns score, PD and up to three risk factors - the reason codes if declined."""
    if bundle is None:
        with open(MODEL_FILE, "rb") as f:
            bundle = pickle.load(f)
    raw = pd.DataFrame([applicant]).astype(float)
    X_row = apply_caps(clean(raw), bundle["caps"])[bundle["features"]]
    p = float(bundle["pipeline"].predict_proba(X_row)[0, 1])
    return {"score": int(pd_to_score(p)), "pd": p,
            "factors": reason_codes(bundle["pipeline"], X_row)}


# ---------------------------------------------------------------------------
# Training and validation
# ---------------------------------------------------------------------------
def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    OUT_DIR.mkdir(exist_ok=True)

    # Step 1 - Data ingestion and cleaning ----------------------------------
    df = clean(pd.read_csv(DATA_FILE))
    X = df.drop(columns=[TARGET] + DROPPED)
    y = df[TARGET]

    line()
    print("CREDIT RISK SCORECARD - LOGISTIC REGRESSION")
    line()
    print(f"Borrowers: {len(df):,} | Features: {X.shape[1]} | Default rate: {y.mean():.2%}")
    print("Missing values (flagged where informative, then filled with training-set median):")
    missing = X.isna().mean()
    for col, pct in missing[missing > 0].sort_values(ascending=False).items():
        print(f"  {col:<42} {pct:6.1%}")
    for flag in FLAG_COLS:
        rate = y[X[flag] == 1].mean()
        print(f"  {flag:<42} default rate when flagged: {rate:.1%}")

    # Step 2 - Stratified 80/20 split BEFORE any fitting (prevents leakage) --
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    # Outlier capping at the training set's 1st / 99th percentiles (not the 0/1 flags)
    caps = {c: (X_train[c].quantile(0.01), X_train[c].quantile(0.99))
            for c in X.columns if c not in FLAG_COLS}
    X_train, X_test = apply_caps(X_train, caps), apply_caps(X_test, caps)

    # Quick multicollinearity check - correlated inputs make coefficients unstable
    corr = X_train.corr().abs()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns)
             for b in corr.columns[i + 1:] if corr.loc[a, b] > 0.7]
    print(f"\nHighly correlated feature pairs (|r| > 0.7): {len(pairs)}")
    for a, b, r in pairs:
        print(f"  {a} <-> {b}: {r:.2f}")

    # Steps 3 & 4 - Imputation + scaling + logistic regression in one pipeline
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=1000)),
    ])
    model.fit(X_train, y_train)

    # Validation - the metrics credit teams use ----------------------------
    pd_test = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pd_test)
    fpr, tpr, _ = roc_curve(y_test, pd_test)
    ks = np.max(tpr - fpr)

    print()
    line()
    print("MODEL VALIDATION (20% hold-out)")
    line()
    print(f"  AUC  : {auc:.3f}")
    print(f"  Gini : {2 * auc - 1:.3f}   (2 x AUC - 1)")
    print(f"  KS   : {ks:.3f}   (max separation of good vs bad)")

    # Calibration by decile: are predicted PDs right in level, not just in ranking?
    deciles = (
        pd.DataFrame({"pd": pd_test, "y": y_test.values,
                      "Decile": pd.qcut(pd_test, 10, labels=range(1, 11))})
        .groupby("Decile", observed=True)
        .agg(Borrowers=("y", "count"), Predicted=("pd", "mean"), Actual=("y", "mean"))
    )
    deciles["Ratio"] = deciles["Predicted"] / deciles["Actual"]

    print()
    line()
    print("CALIBRATION BY DECILE (1 = safest 10%, 10 = riskiest 10%)")
    line()
    print(f"{'Decile':<8}{'Borrowers':>11}{'Predicted PD':>15}{'Actual rate':>14}{'Pred / Actual':>16}")
    line("-")
    for d, r in deciles.iterrows():
        print(f"{d:<8}{int(r['Borrowers']):>11,}{r['Predicted']:>15.2%}"
              f"{r['Actual']:>14.2%}{r['Ratio']:>16.2f}")
    line("-")
    print(f"{'All':<8}{len(pd_test):>11,}{pd_test.mean():>15.2%}{y_test.mean():>14.2%}")
    print("Pred / Actual above 1 = model overstates risk; below 1 = understates it.")

    # Step 5 - Risk Weight Matrix -------------------------------------------
    coefs = model.named_steps["logit"].coef_[0]
    matrix = pd.DataFrame({"Feature": X.columns, "Coefficient": coefs})
    matrix["Odds_x_per_1SD"] = np.exp(matrix["Coefficient"])
    matrix["Direction"] = np.where(matrix["Coefficient"] > 0, "RAISES risk", "lowers risk")
    matrix = matrix.reindex(matrix["Coefficient"].abs().sort_values(ascending=False).index)
    matrix.insert(0, "Rank", range(1, len(matrix) + 1))

    print()
    line()
    print("RISK WEIGHT MATRIX (standardized coefficients, ranked by impact)")
    line()
    print(f"{'Rank':<5}{'Feature':<42}{'Coef':>8}{'Odds x':>9}  Direction")
    line("-")
    for _, r in matrix.iterrows():
        print(f"{r['Rank']:<5}{r['Feature']:<42}{r['Coefficient']:>+8.3f}"
              f"{r['Odds_x_per_1SD']:>9.2f}  {r['Direction']}")
    line("-")
    print("Odds x = how default odds multiply for a 1 standard-deviation increase.")

    # Score conversion: PD -> 300-850 ---------------------------------------
    scores = pd_to_score(pd_test)
    bands = pd.cut(scores, bins=[299, 579, 669, 739, 799, 850],
                   labels=["Poor 300-579", "Fair 580-669", "Good 670-739",
                           "V.Good 740-799", "Excep 800-850"])
    band_table = (
        pd.DataFrame({"Band": bands, "pd": pd_test, "Default": y_test.values})
        .groupby("Band", observed=False)
        .agg(Borrowers=("Default", "count"), Predicted=("pd", "mean"),
             Default_Rate=("Default", "mean"))
    )

    print()
    line()
    print(f"CREDIT SCORE BANDS (hold-out) | {BASE_SCORE} pts = {BASE_ODDS}:1 odds, +{PDO} pts doubles odds")
    line()
    print(f"{'Score band':<16}{'Borrowers':>11}{'Share':>9}{'Predicted PD':>15}{'Default rate':>15}")
    line("-")
    for band, r in band_table.iterrows():
        if not r["Borrowers"]:
            print(f"{band:<16}{0:>11}{'-':>9}{'-':>15}{'-':>15}")
            continue
        print(f"{band:<16}{int(r['Borrowers']):>11,}{r['Borrowers'] / len(scores):>9.1%}"
              f"{r['Predicted']:>15.2%}{r['Default_Rate']:>15.2%}")
    line("-")

    # Expected Loss: EL = PD x LGD x EAD (2-year horizon, matching the target)
    utilization = X_test["RevolvingUtilizationOfUnsecuredLines"].clip(0, 1).values
    drawn = utilization * CREDIT_LIMIT
    ead = drawn + CCF * (CREDIT_LIMIT - drawn)          # drawn + CCF x unused limit
    expected_loss = pd_test * LGD * ead

    el_table = (
        pd.DataFrame({"Band": bands, "EAD": ead, "EL": expected_loss})
        .groupby("Band", observed=False)
        .agg(Borrowers=("EL", "count"), Total_EAD=("EAD", "sum"),
             Total_EL=("EL", "sum"), Avg_EL=("EL", "mean"))
    )
    total_ead, total_el = ead.sum(), expected_loss.sum()

    print()
    line()
    print("EXPECTED LOSS (hold-out portfolio)  |  EL = PD x LGD x EAD")
    line()
    print(f"Assumptions: ${CREDIT_LIMIT:,} limit per borrower, CCF {CCF:.0%}, LGD {LGD:.0%}")
    print(f"{'Score band':<16}{'Exposure':>14}{'Expected loss':>16}{'Avg EL':>10}{'Loss rate':>11}")
    line("-")
    for band, r in el_table.iterrows():
        if not r["Borrowers"]:
            continue
        print(f"{band:<16}{r['Total_EAD']:>14,.0f}{r['Total_EL']:>16,.0f}"
              f"{r['Avg_EL']:>10,.0f}{r['Total_EL'] / r['Total_EAD']:>11.2%}")
    line("-")
    print(f"{'Portfolio':<16}{total_ead:>14,.0f}{total_el:>16,.0f}"
          f"{total_el / len(ead):>10,.0f}{total_el / total_ead:>11.2%}")
    poor_share = el_table.loc["Poor 300-579", "Total_EL"] / total_el
    print(f"\nThe Poor band holds {(bands == 'Poor 300-579').mean():.0%} of borrowers "
          f"but {poor_share:.0%} of expected loss.")

    # Save the model bundle -------------------------------------------------
    bundle = {
        "pipeline": model,
        "caps": caps,
        "features": list(X.columns),
        "score_params": {"base_score": BASE_SCORE, "base_odds": BASE_ODDS, "pdo": PDO},
        "el_assumptions": {"credit_limit": CREDIT_LIMIT, "ccf": CCF, "lgd": LGD},
        "metrics": {"auc": auc, "gini": 2 * auc - 1, "ks": ks},
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(bundle, f)
    matrix.to_csv(MATRIX_FILE, index=False)

    # Scoring new applicants with reason codes ------------------------------
    examples = {
        "Maria": {"RevolvingUtilizationOfUnsecuredLines": 0.95, "age": 28,
                  "NumberOfTime30-59DaysPastDueNotWorse": 3,
                  "MonthlyIncome": 6000, "NumberOfOpenCreditLinesAndLoans": 8,
                  "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 0,
                  "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 0},
        "James": {"RevolvingUtilizationOfUnsecuredLines": 0.10, "age": 52,
                  "NumberOfTime30-59DaysPastDueNotWorse": 0,
                  "MonthlyIncome": 4500, "NumberOfOpenCreditLinesAndLoans": 6,
                  "NumberOfTimes90DaysLate": 0, "NumberRealEstateLoansOrLines": 1,
                  "NumberOfTime60-89DaysPastDueNotWorse": 0, "NumberOfDependents": 2},
    }
    print()
    line()
    print("SCORING NEW APPLICANTS (score_applicant)")
    print("Risk factors = what pushes risk above average; if declined, these are the reason codes")
    line()
    for name, applicant in examples.items():
        result = score_applicant(applicant, bundle)
        print(f"{name}: score {result['score']} | PD {result['pd']:.1%}")
        for k, factor in enumerate(result["factors"], 1):
            print(f"   Risk factor {k}: {factor}")
        if not result["factors"]:
            print("   No factors raise risk above the average borrower")

    # Charts ----------------------------------------------------------------
    ink, muted, grid, blue = "#0b0b0b", "#898781", "#e1e0d9", "#2a78d6"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ROC curve with the KS point marked
    i_ks = int(np.argmax(tpr - fpr))
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    ax.plot([0, 1], [0, 1], color=muted, lw=1, ls="--", label="Random guessing")
    ax.plot(fpr, tpr, color=blue, lw=2, label=f"Model (AUC {auc:.3f})")
    ax.plot([fpr[i_ks]] * 2, [fpr[i_ks], tpr[i_ks]], color=ink, lw=1)
    ax.annotate(f"KS = {ks:.3f}", (fpr[i_ks], (fpr[i_ks] + tpr[i_ks]) / 2),
                xytext=(8, 0), textcoords="offset points", va="center", color=ink)
    ax.set(xlabel="False positive rate (good borrowers flagged)",
           ylabel="True positive rate (defaulters caught)", xlim=(0, 1), ylim=(0, 1))
    ax.set_title("ROC curve - 20% hold-out", loc="left", color=ink)
    ax.grid(color=grid, lw=0.6)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc_curve.png")
    plt.close(fig)

    # Calibration: predicted vs actual default rate by decile (log scale)
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=150)
    lims = [0.002, 0.6]
    ax.plot(lims, lims, color=muted, lw=1, ls="--", label="Perfect calibration")
    ax.plot(deciles["Predicted"], deciles["Actual"], color=blue, lw=2,
            marker="o", ms=7, label="Model, by decile")
    for d, r in deciles.iterrows():
        if d in (1, 10):
            ax.annotate(f"Decile {d}", (r["Predicted"], r["Actual"]),
                        xytext=(8, -4), textcoords="offset points", color=ink)
    ax.set(xscale="log", yscale="log", xlim=lims, ylim=lims,
           xlabel="Average predicted PD", ylabel="Actual default rate")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
    ax.set_title("Calibration - predicted vs actual default rate", loc="left", color=ink)
    ax.grid(color=grid, lw=0.6, which="major")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "calibration.png")
    plt.close(fig)

    print(f"\nSaved to outputs/: {MODEL_FILE.name}, {MATRIX_FILE.name}, roc_curve.png, calibration.png")


if __name__ == "__main__":
    main()
