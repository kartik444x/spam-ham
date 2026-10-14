"""
Model Training Pipeline
-----------------------
Trains and evaluates multiple spam-detection classifiers, then saves
the best model artefacts to the  models/  directory.

Fixes applied vs. original project1.py
---------------------------------------
* x_test_tfidf now uses  .transform()  instead of  .fit_transform()
  (the original caused data-leakage and an out-of-vocabulary mismatch).
* Removed duplicate imports (StandardScaler, WordNetLemmatizer).
* All code is inside a proper  if __name__ == '__main__':  guard.
* Models are persisted with joblib so the Flask app can load them.
* Confusion-matrix plot is saved to disk instead of blocking plt.show().
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — no GUI window
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from src.preprocessor import preprocess_text

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, 'spam.csv')
MODEL_DIR  = os.path.join(BASE_DIR, 'models')
PLOT_DIR   = os.path.join(BASE_DIR, 'static', 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR,  exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    """Load the SMS spam dataset and add a binary label column."""
    df = pd.read_csv(path, encoding='latin-1')
    df = df[['v1', 'v2']].rename(columns={'v1': 'label', 'v2': 'text'})
    df['is_spam'] = df['label'].map({'ham': 0, 'spam': 1})
    return df


def build_pipeline(df: pd.DataFrame, method: str = 'stem',
                   ngram_range: tuple = (1, 2), max_features: int = 3000):
    """
    Vectorise cleaned text and return train/test splits.

    Parameters
    ----------
    df           : DataFrame with 'text' and 'is_spam' columns.
    method       : Preprocessing method ('stem' or 'lemma').
    ngram_range  : n-gram range for TF-IDF.
    max_features : Vocabulary size cap.

    Returns
    -------
    x_train_vec, x_test_vec, y_train, y_test, vectorizer
    """
    col = 'clean_stem' if method == 'stem' else 'clean_lem'

    if col not in df.columns:
        print(f'  Preprocessing text with method={method} …')
        df[col] = df['text'].apply(lambda t: preprocess_text(t, method=method))

    x_train, x_test, y_train, y_test = train_test_split(
        df[col], df['is_spam'],
        test_size=0.20, random_state=42, stratify=df['is_spam']
    )

    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec  = vectorizer.transform(x_test)   # ← .transform() only, no leakage

    return x_train_vec, x_test_vec, y_train, y_test, vectorizer


def evaluate_model(model, x_test, y_test, name: str) -> dict:
    """Return a dict of key metrics for a fitted model."""
    y_pred   = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred,
                                     target_names=['ham', 'spam'],
                                     output_dict=True)
    print(f'\n{"-"*60}')
    print(f'  {name}')
    print(f'  Accuracy : {accuracy * 100:.2f}%')
    print(f'  Precision (spam): {report["spam"]["precision"]:.4f}')
    print(f'  Recall    (spam): {report["spam"]["recall"]:.4f}')
    print(f'  F1-score  (spam): {report["spam"]["f1-score"]:.4f}')
    return {'accuracy': accuracy, 'report': report, 'y_pred': y_pred}


def save_confusion_matrix(model, x_test, y_test, name: str, filename: str):
    """Save a confusion-matrix figure to static/plots/."""
    cm   = confusion_matrix(y_test, model.predict(x_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ham', 'Spam'])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix — {name}', fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f'  Plot saved -> {path}')


def train_and_compare(df: pd.DataFrame):
    """
    Full experiment:
      1. Stemming  + Naive Bayes  (1-gram)
      2. Stemming  + Naive Bayes  (1&2-gram)
      3. Lemmatise + Naive Bayes  (1&2-gram)
      4. Stemming  + Logistic Regression (1&2-gram)  ← usually best
    Save the winner (Logistic Regression) for the Flask app.
    """
    results = {}

    # ── 1. NB – unigrams ───────────────────────────────────────────────────
    print('\n[1/4] Naive Bayes – Unigrams (stem)')
    xtr, xte, ytr, yte, vec1 = build_pipeline(df, method='stem', ngram_range=(1, 1))
    nb1 = MultinomialNB().fit(xtr, ytr)
    results['NB_1gram'] = evaluate_model(nb1, xte, yte, 'NaiveBayes 1-gram')
    save_confusion_matrix(nb1, xte, yte, 'NaiveBayes 1-gram', 'cm_nb_1gram.png')

    # ── 2. NB – bigrams ────────────────────────────────────────────────────
    print('\n[2/4] Naive Bayes – Bigrams (stem)')
    xtr, xte, ytr, yte, vec2 = build_pipeline(df, method='stem', ngram_range=(1, 2))
    nb2 = MultinomialNB().fit(xtr, ytr)
    results['NB_bigram'] = evaluate_model(nb2, xte, yte, 'NaiveBayes 1+2-gram')
    save_confusion_matrix(nb2, xte, yte, 'NaiveBayes 1+2-gram', 'cm_nb_bigram.png')

    # ── 3. NB – lemmatise ──────────────────────────────────────────────────
    print('\n[3/4] Naive Bayes – Bigrams (lemma)')
    xtr_l, xte_l, ytr_l, yte_l, vec_l = build_pipeline(df, method='lemma', ngram_range=(1, 2))
    nb_l = MultinomialNB().fit(xtr_l, ytr_l)
    results['NB_lemma'] = evaluate_model(nb_l, xte_l, yte_l, 'NaiveBayes Lemma')
    save_confusion_matrix(nb_l, xte_l, yte_l, 'NaiveBayes Lemma', 'cm_nb_lemma.png')

    # ── 4. Logistic Regression ─────────────────────────────────────────────
    print('\n[4/4] Logistic Regression – Bigrams (stem)')
    xtr2, xte2, ytr2, yte2, vec_lr = build_pipeline(df, method='stem', ngram_range=(1, 2))
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(xtr2, ytr2)
    results['LR'] = evaluate_model(lr, xte2, yte2, 'Logistic Regression')
    save_confusion_matrix(lr, xte2, yte2, 'LogisticRegression', 'cm_lr.png')

    # ── Summary ────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print('  MODEL COMPARISON SUMMARY')
    print('='*60)
    for name, r in results.items():
        print(f'  {name:<20}  accuracy={r["accuracy"]*100:.2f}%')

    # ── Save best model (pick dynamically by accuracy) ─────────────────────
    model_map = {
        'NB_1gram':  (nb1,  vec1),
        'NB_bigram': (nb2,  vec2),
        'NB_lemma':  (nb_l, vec_l),
        'LR':        (lr,   vec_lr),
    }
    best_name  = max(results, key=lambda k: results[k]['accuracy'])
    best_model, best_vec = model_map[best_name]
    print(f'\n  Best model: {best_name} ({results[best_name]["accuracy"]*100:.2f}%)')
    joblib.dump(best_model, os.path.join(MODEL_DIR, 'spam_classifier.pkl'))
    joblib.dump(best_vec,   os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    print('  Saved to  models/')


    return results


if __name__ == '__main__':
    print('Loading dataset …')
    df = load_data(DATA_PATH)
    print(f'  Rows: {df.shape[0]}  |  Columns: {df.shape[1]}')
    print(df['label'].value_counts().to_string())

    train_and_compare(df)
    print('\nTraining complete.')
