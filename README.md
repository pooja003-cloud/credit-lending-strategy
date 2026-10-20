# Credit Lending Strategy — Scorecard vs XGBoost

Fintech lenders say they can safely approve more borrowers than traditional banks. This project tests that claim among borrowers who all received credit.

**What I set out to do.** Challenge my bank-style scorecard with XGBoost on exactly the same borrowers, and measure the difference in lending decisions, not just accuracy.

**What I did**
- Trained XGBoost with monotone constraints, so every factor moves risk in a direction a lender can explain (cost: 0.002 AUC)
- Showed why the common fix for rare defaults is harmful: class weighting made predicted default rates 4.7× too high
- Compared approval rate, default rate and profit at every possible approval cut-off
- Explained declines with SHAP reason codes, which US lending rules require

**Results**
- **AUC 0.861 → 0.868**, with better-calibrated probabilities
- **At a 2% target default rate, XGBoost approves about 4% more applicants** (+2,907 per 100,000 on unseen borrowers, in my validation project)
- **Profit uplift of +5% to +37%, depending on how much approved borrowers borrow**; most of it comes from a balance assumption, not the model alone
- **The borrowers XGBoost approves differently** are heavy credit users with clean payment histories, defaulting at 3.07% vs 6.86% for those only the scorecard approves (at 70% approval)
- No advantage for a lender that approves nearly everyone

<p align="center"><img src="outputs/approval_vs_default.png" width="55%" alt="Default rate among approved borrowers by approval rate, scorecard vs XGBoost"></p>

**How it connects.** This expands a smaller project from my Financial Data and FinTech course (MS in Quantitative Finance). It challenges my [credit risk scorecard](https://github.com/pooja003-cloud/credit-risk-scorecard), and my [validation](https://github.com/pooja003-cloud/credit-model-validation) project re-tests its results.

**Run it.** Download `cs-training.csv` from [Kaggle's Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data), save it as `data/credit_data.csv`, then run `pip install -r requirements.txt` and `python strategy.py`.

**Full details:** methods, all tables, assumptions and limitations are in [REPORT.md](REPORT.md).
