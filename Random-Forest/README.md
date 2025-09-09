# Random Forest — Titanic

Predict Titanic survival with a RandomForestClassifier.

## Run
```powershell
cd Random-Forest
python .\titanic.py
```

## Details
- Loads `titanic` from seaborn and drops rows missing key fields
- Encodes categorical features (`sex`, `embarked`) with LabelEncoder
- Train/test split with stratification
- Prints accuracy, classification report, and confusion matrix
