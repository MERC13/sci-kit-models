# K-Means Clustering examples

Two small clustering demos using scikit-learn K-Means.

## 1) Breast Cancer (scikit-learn)
- Loads `load_breast_cancer` dataset
- Standardizes features, K=2
- Evaluates with silhouette score and contingency vs true labels
- Visualizes clusters in PCA(2) space

Run:
```powershell
cd K-Means-Clustering
python .\breast_cancer.py
```

## 2) Palmer Penguins (seaborn)
- Loads `penguins` via seaborn; uses bill and flipper measurements
- Standardizes features, K=3
- Prints unscaled cluster centers and counts
- Visualizes a 2D scatter colored by predicted cluster

Run:
```powershell
cd K-Means-Clustering
python .\penguins.py
```

Notes:
- First seaborn dataset load downloads the CSV
- Figures open in a window; run in a desktop session
