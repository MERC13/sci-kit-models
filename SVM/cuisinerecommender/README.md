# SVM — cuisine recommender (ONNX export)

Train a linear SVM classifier on a feature-engineered cuisine dataset, evaluate, and export to ONNX.

## Data
- Input CSV: `cleaned_cuisines.csv` in this folder
- Features: columns from index 2 onward
- Target: `cuisine`

## Run
```powershell
cd SVM\cuisinerecommender
pip install -r ..\..\requirements.txt
python .\main.py
```

## Outputs
- `model.onnx` — exported ONNX model via skl2onnx
- Console — classification report for hold-out test set

Notes:
- `skl2onnx` requires compatible scikit-learn versions (see requirements.txt)
- The input tensor shape in export is set to 380 features; adjust if your CSV schema differs
