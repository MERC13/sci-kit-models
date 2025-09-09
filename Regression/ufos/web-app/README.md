# UFOs web app

Minimal Flask app that loads `ufo-model.pkl` and predicts likely country from user input.

## Run
```powershell
cd Regression\ufos\web-app
pip install -r ..\..\..\requirements.txt
python .\app.py
```

Open http://127.0.0.1:5001/ in a browser.

If you don’t have a model yet, train one first (see `../README.md`).
