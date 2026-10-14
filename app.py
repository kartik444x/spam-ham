"""
Flask Web Application — SMS Spam Detector
==========================================
Run:
    python app.py

Then open  http://127.0.0.1:5000  in your browser.
"""

import os
import joblib
from flask import Flask, render_template, request, jsonify

from src.preprocessor import preprocess_text

# ── App setup ──────────────────────────────────────────────────────────────
app     = Flask(__name__)
BASE    = os.path.dirname(os.path.abspath(__file__))
MODEL_P = os.path.join(BASE, 'models', 'spam_classifier.pkl')
VEC_P   = os.path.join(BASE, 'models', 'tfidf_vectorizer.pkl')

# Load model & vectoriser once at startup
model      = None
vectorizer = None

def load_model():
    """Load persisted artefacts from  models/ ."""
    global model, vectorizer
    if os.path.exists(MODEL_P) and os.path.exists(VEC_P):
        model      = joblib.load(MODEL_P)
        vectorizer = joblib.load(VEC_P)
        print('[OK] Model & vectorizer loaded.')
    else:
        print('[WARN] Model files not found. Run  python train.py  first.')


# ── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    model_ready = model is not None and vectorizer is not None
    return render_template('index.html', model_ready=model_ready)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts JSON: { "message": "..." }
    Returns JSON: { "label": "spam"|"ham", "confidence": float, "processed": str }
    """
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded. Run python train.py first.'}), 503

    data    = request.get_json(force=True)
    message = data.get('message', '').strip()

    if not message:
        return jsonify({'error': 'No message provided.'}), 400

    cleaned   = preprocess_text(message, method='lemma')
    vec_input = vectorizer.transform([cleaned])
    pred      = model.predict(vec_input)[0]
    proba     = model.predict_proba(vec_input)[0]
    confidence = float(proba[pred]) * 100

    return jsonify({
        'label':      'spam' if pred == 1 else 'ham',
        'confidence': round(confidence, 2),
        'processed':  cleaned,
    })


@app.route('/batch', methods=['POST'])
def batch_predict():
    """
    Accepts JSON: { "messages": ["msg1", "msg2", ...] }
    Returns JSON: list of prediction objects
    """
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model not loaded.'}), 503

    data     = request.get_json(force=True)
    messages = data.get('messages', [])

    if not messages:
        return jsonify({'error': 'No messages provided.'}), 400

    results = []
    for msg in messages:
        cleaned   = preprocess_text(msg, method='lemma')
        vec_input = vectorizer.transform([cleaned])
        pred      = model.predict(vec_input)[0]
        proba     = model.predict_proba(vec_input)[0]
        confidence = float(proba[pred]) * 100
        results.append({
            'message':    msg,
            'label':      'spam' if pred == 1 else 'ham',
            'confidence': round(confidence, 2),
        })

    return jsonify(results)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == '__main__':
    load_model()
    app.run(debug=True, port=5000)
