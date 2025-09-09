# scikit-learn models portfolio

This repo collects small, focused examples of classic ML with scikit-learn: classification, regression, and clustering. Each folder is a self-contained script or mini-app with minimal setup.

## Quick start

Use a virtual environment and install the pinned dependencies.

```powershell
# Clone
git clone https://github.com/MERC13/sci-kit-models.git
cd sci-kit-models

# Create & activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate

# Install deps
pip install -r requirements.txt
```

Then open any folder below and follow its README for run steps.

## What’s inside

- banking example — Bank marketing classification with multiple models and pipelines.
- K-Means-Clustering — K-Means on Breast Cancer and Palmer Penguins datasets.
- KNN — K-Nearest Neighbors on the Wine dataset.
- Naive-Bayes — Multinomial NB on 20 Newsgroups (TF–IDF).
- Random-Forest — Titanic survival prediction with a Random Forest.
- Regression/ufos — Train a simple logistic regression and a tiny Flask web app.
- SVM/cuisinerecommender — Linear SVM trained on cuisines, exports ONNX.

See each folder’s README for: dataset, how to run, and outputs/screens.

## Notes

- Python 3.10+ recommended. Datasets fetched via scikit-learn or seaborn will download on first run.
- Some examples visualize with matplotlib/seaborn; running in a headful environment is recommended.

