# Customer Churn Prediction

Predict whether a telecom customer will leave (`Churn` = Yes/No) using the IBM Telco Customer Churn dataset
(7,043 rows, 19 features; 7,021 rows after removing 22 exact duplicates; 26.4% churn). Built with pandas, matplotlib and scikit-learn.

## Folder structure and how to run

```
customer-churn-ml/
├── data/telco.csv             # IBM Telco Customer Churn dataset (included)
├── churn_project.ipynb        # step-by-step notebook with explanations (already executed)
├── churn_project.py           # same workflow as a runnable script
├── reports/
│   ├── model_comparison.csv   # validation results for all 4 models
│   ├── final_test_results.csv # one-time test result of the chosen model
│   └── figures/               # 5 charts
├── requirements.txt           # pinned versions
└── README.md
```

```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python churn_project.py        # ~40 seconds; works from any folder
# or: jupyter notebook churn_project.ipynb   (Run All)
```

**Reproducibility:** the numbers below were produced with Python 3.13, pandas 3.0.5, numpy 2.5.3, matplotlib 3.11.2 and
scikit-learn 1.9.1 (pinned in `requirements.txt`), random seed 42. The script and the notebook were each run in a clean
virtual environment with these versions and give identical CSVs; running the script twice also gives identical results.
Other library versions can change tree and forest results slightly (tie-breaking, random number handling), so if you
change versions, re-run and use your own printed numbers. The data is the public "Telco Customer Churn" CSV from IBM / Kaggle.

## Workflow

| Step | What happens | Concept shown |
|---|---|---|
| 1 | Load data, fix `TotalCharges` (text -> number), drop `customerID`, remove exact duplicates, define X and y, 3 EDA charts | Features / target, supervised learning |
| 2 | Stratified 70 / 15 / 15 split **before** any preprocessing | Train / validation / test, leakage |
| 3 | Pipeline: impute -> scale / one-hot -> Logistic Regression baseline | Preprocessing, baseline |
| 4 | Logistic Regression vs Decision Tree vs Random Forest, train vs validation F1 | Model comparison, overfitting |
| 5a | Sweep `C` (logistic regression) and `max_depth` (tree) | Regularization |
| 5b | `GridSearchCV` on Random Forest, 5-fold stratified CV, training data only, metric = F1 | Cross-validation, tuning |
| 6 | Pick model by validation F1; pick probability threshold on validation | Model selection |
| 7 | Evaluate once on test: confusion matrix, report, ROC curve | Final evaluation |
| 8 | Logistic Regression coefficients, Random Forest importances | Interpretability |
| Extra A | Linear regression predicting `MonthlyCharges` (optional) | Linear regression |
| Extra B | KMeans on tenure + monthly charges, no labels (optional) | Unsupervised learning |

## Results

Split sizes: train 4,914 / validation 1,053 / test 1,054 rows (churn rate about 26.5% in each).

**Validation set** (`reports/model_comparison.csv`):

| Model | Train F1 | Val F1 | Gap | Val precision | Val recall | Val ROC-AUC |
|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.634 | **0.628** | 0.007 | 0.520 | 0.791 | 0.843 |
| Decision Tree (no limit) | 0.996 | 0.495 | 0.501 | 0.504 | 0.486 | 0.657 |
| Random Forest (default) | 0.987 | 0.588 | 0.399 | 0.558 | 0.622 | 0.820 |
| Random Forest (tuned) | 0.721 | 0.626 | 0.095 | 0.533 | 0.759 | 0.846 |

**Tuning (5-fold CV on the training set, F1):** these are *tuning scores*, because the same folds were used to choose the
parameters, so the tuned score is slightly optimistic. The independent check is the validation column above.
Default Random Forest 0.592 -> best grid setting 0.633 (`max_depth=10, min_samples_leaf=5, n_estimators=200`); the tuned setting was higher in all 5 folds.
On the validation set tuning raised F1 from 0.588 to 0.626 and cut the train-validation gap from 0.399 to 0.095.

**Final model: Logistic Regression**, threshold 0.50 (validation showed no better threshold).
**Test set (used once):** precision 0.506, recall 0.814, F1 0.624, ROC-AUC 0.852.
Confusion matrix: 553 true negatives, 222 false positives, 52 false negatives, 227 true positives.

**Why this model?** The rule was fixed in advance: highest validation F1. Logistic Regression scored 0.628 and the tuned
Random Forest 0.626, so this is effectively a tie (the tuned forest even has a slightly higher validation ROC-AUC, 0.846 vs 0.843).
When two models perform the same, the simpler and more interpretable one is the sensible choice. Do not claim the forest
is worse or better; claim that tuning rescued it from overfitting to the level of the linear model, and that the churn signal is mostly simple.

## Metric interpretation

- **Accuracy is misleading here:** always predicting "no churn" is 73.6% accurate and finds zero churners.
- **Precision** = of customers we flag, how many really leave (avoid wasting retention offers).
- **Recall** = of customers who really leave, how many we catch (avoid losing customers silently). Our final model catches about 81% but only about half of the flagged customers actually leave.
- **F1** balances the two and was our primary metric, chosen before tuning.
- **ROC-AUC** (0.85) is threshold-independent: it shows the model ranks churners above non-churners well.
- **Overfitting evidence:** the unlimited tree scores 0.996 on train but 0.495 on validation. Limiting depth to 3 lifts validation F1 to 0.624 (see `02_overfitting_curves.png`).
- **Logistic Regression `C`:** validation F1 only moves between 0.621 and 0.628 across a 100,000x range of `C`, so regularization matters little for this simple linear model. The tree curve is the clearer demonstration.

## Optional extras (separate from the churn evaluation)

- **Linear regression** predicts `MonthlyCharges` with R² = 0.949. This is an easy problem, not an impressive result: the monthly price is mostly determined by the services a customer has.
- **KMeans** groups customers by tenure and monthly charges (three segments). It scales and clusters all rows, so it is a descriptive demo and is *not* part of the leakage-free evaluation. The newer-customer, higher-charge group (avg 12.6 months, $75/month) churns at 47% versus 12-16% for the others.

## Leakage checklist

- [x] Data is split before any preprocessing is fitted.
- [x] Imputer, scaler and encoder are inside the Pipeline, so they learn only from training data.
- [x] Cross-validation re-fits preprocessing inside each fold.
- [x] `customerID` dropped; no feature is recorded after a customer leaves (column meanings checked).
- [x] Test data is not used for model selection, tuning or the threshold.
- [x] Duplicates: 0 repeated IDs; 22 rows identical in every feature and the target were removed before splitting, so a copy cannot sit in train while its twin is in test.
- [x] The tuning CV score is labelled as a tuning score; model comparison uses the untouched validation set.

## Limitations

- F1 ≈ 0.62 is moderate; the dataset has only basic customer attributes (no usage or complaint history).
- One random split: scores shift slightly with the seed, and a 0.002 validation F1 difference between two models is within noise.
- The threshold maximizes F1, not real business cost (a retention offer vs a lost customer).
- Models were not retrained on train + validation before testing (kept simple).
- Rows that are near-duplicates but not identical were not removed.
- Results can differ slightly with other library versions (see Reproducibility).
