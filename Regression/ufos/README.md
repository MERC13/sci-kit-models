# Regression/UFOS — simple model + Flask app

Train a small logistic regression model to predict likely country from UFO sighting duration and location, then serve a minimal Flask form to make predictions.

## Train the model
```powershell
cd Regression\ufos

# Train and export ufo-model.pkl
python .\main.py
```

Outputs: `ufo-model.pkl` in the same folder.

## Run the web app
The web app reads a model file named `ufo-model.pkl` from the `web-app` folder. Copy the trained file or retrain from that folder.

```powershell
cd Regression\ufos\web-app

# Option A: if you already have a trained model in this folder
python .\app.py

# Option B: train here so the pickle lands next to app.py
# (optional) python ..\main.py
# python .\app.py
```

Then open http://127.0.0.1:5001/ in your browser.

Notes:
- The form expects integers for seconds, latitude, longitude; the model was trained on filtered ranges (1–60 seconds).
- Country index is mapped to: Australia, Canada, Germany, UK, US.
