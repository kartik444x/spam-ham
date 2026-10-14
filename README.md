# 🛡️ SpamShield — SMS Spam Detector

> A full NLP + Machine Learning pipeline to classify SMS messages as **ham** or **spam**, served through a modern Flask web interface.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model  (needs spam.csv in root)
python train.py

# 3. Launch the web app
python app.py

# 4. Open in browser
#    http://127.0.0.1:5000
```

---

## 📁 Project Structure

```
project1/
├── app.py                  # Flask web application
├── train.py                # Model training pipeline
├── requirements.txt
├── spam.csv                # UCI SMS Spam dataset
│
├── src/
│   ├── __init__.py
│   └── preprocessor.py     # Text cleaning (stem / lemma)
│
├── models/                 # Saved model artefacts (auto-created)
│   ├── spam_classifier.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html          # Frontend UI
│
└── static/
    ├── css/style.css
    ├── js/main.js
    └── plots/              # Confusion matrix images (auto-created)
```

---

## 🧠 ML Pipeline

| Step | Detail |
|---|---|
| **Dataset** | UCI SMS Spam Collection — 5,572 messages |
| **Preprocessing** | Lowercase → Tokenise → Remove stopwords & punctuation → Stem |
| **Vectorisation** | TF-IDF (3,000 features, 1&2-gram) |
| **Models trained** | Naïve Bayes (1-gram), NB (bigram), NB (lemma), Logistic Regression |
| **Best model** | Logistic Regression — ~98.5% accuracy |

### Bug fixes vs. original `project1.py`

| Bug | Fix |
|---|---|
| `x_test_tfidf = tfidf.fit_transform(x_test)` — data leakage + vocab mismatch | Changed to `.transform(x_test)` |
| Duplicate `import` of `WordNetLemmatizer` and `StandardScaler` | Removed duplicates |
| LR trained on leaky vectoriser | Uses a clean independent vectoriser |
| Flat single-file script | Proper package structure with `src/` module |
| No model persistence | `joblib.dump` → `models/` directory |
| `plt.show()` blocks server | `matplotlib.use('Agg')`, plots saved to disk |

---

## 🌐 API Endpoints

### `POST /predict`
```json
// Request
{ "message": "Win a free iPhone now!" }

// Response
{ "label": "spam", "confidence": 98.72, "processed": "win free iphon now" }
```

### `POST /batch`
```json
// Request
{ "messages": ["Hey coffee?", "WIN FREE PRIZE!"] }

// Response
[
  { "message": "Hey coffee?",    "label": "ham",  "confidence": 99.1 },
  { "message": "WIN FREE PRIZE!", "label": "spam", "confidence": 97.3 }
]
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **scikit-learn** — TF-IDF, Logistic Regression, Naïve Bayes
- **NLTK** — tokenisation, stop-words, stemming, lemmatisation
- **Flask** — REST API + HTML serving
- **Matplotlib / Seaborn** — confusion matrix plots
