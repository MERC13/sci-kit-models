# Banking example — deposit classification

Train multiple classifiers on the UCI bank marketing dataset (via a hosted CSV). Pipelines handle preprocessing; metrics and ROC are plotted. Saves a Random Forest model and a full-dataset prediction CSV.

## Dataset
- Source CSV: https://raw.githubusercontent.com/rafiag/DTI2020/main/data/bank.csv
- Target: `deposit` (yes/no)
- Drops column: `duration`

## How to run

```powershell
# From repo root
cd "banking example"

# Run training
python .\train.py
```

## Outputs
- `bank_deposit_classification.joblib` — persisted Random Forest pipeline
- `deposit_prediction.csv` — original rows plus `deposit_prediction`
- Console: metric summary and preview of predictions
- Plots: bar chart of metrics and ROC curves per model

## Models compared
- Decision Tree, Random Forest, GaussianNB, KNN(5)
- Preprocessing: StandardScaler (numeric) + OneHotEncoder (categorical)
