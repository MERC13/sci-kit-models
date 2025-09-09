# Support Vector Machines (SVM)

This folder contains SVM-based examples. Each subproject documents data, training, and outputs.

## Projects

### 1) Cuisine recommender (ONNX export)
- Data: `SVM/cuisinerecommender/cleaned_cuisines.csv`
- Model: linear SVM (probability enabled), exports to ONNX
- Output: `model.onnx` and a classification report

Run:
```powershell
cd SVM\cuisinerecommender
python .\main.py
```

Notes:
- Adjust the ONNX input feature size if the CSV schema changes.
- `skl2onnx` versions must be compatible with your scikit-learn version.
