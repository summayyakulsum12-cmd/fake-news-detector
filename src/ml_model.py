"""
Loads the trained TF-IDF + LogisticRegression model and exposes a simple
predict() function used by the hybrid pipeline.
"""
import os
import joblib

from src.data_preprocessing import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

_vectorizer = None
_model = None


def _load():
    global _vectorizer, _model
    if _vectorizer is None or _model is None:
        vec_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
        model_path = os.path.join(MODELS_DIR, "ml_model.pkl")
        if not (os.path.exists(vec_path) and os.path.exists(model_path)):
            raise FileNotFoundError(
                "ML model artifacts not found. Run `python src/train_ml_model.py` first."
            )
        _vectorizer = joblib.load(vec_path)
        _model = joblib.load(model_path)
    return _vectorizer, _model


def predict(text: str) -> dict:
    """Returns {'label': 'FAKE'|'REAL', 'confidence': float 0-1, 'proba': {...}}"""
    vectorizer, model = _load()
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    label = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    confidence = float(proba[label])
    return {
        "label": label,
        "confidence": round(confidence, 4),
        "proba": {k: round(float(v), 4) for k, v in proba.items()},
    }


def is_ready() -> bool:
    try:
        _load()
        return True
    except FileNotFoundError:
        return False
