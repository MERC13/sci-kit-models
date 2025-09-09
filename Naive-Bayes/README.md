# Naive Bayes — 20 Newsgroups

Multinomial Naive Bayes with TF–IDF features on a subset of the 20 Newsgroups corpus.

## Categories
- alt.atheism, comp.graphics, sci.space, talk.religion.misc

## Run
```powershell
cd Naive-Bayes
python .\newsgroups.py
```

## What it does
- Downloads the subset (first run)
- Vectorizes text with TfidfVectorizer (English stopwords)
- Trains/test-splits with stratification
- Prints accuracy, classification report and confusion matrix
