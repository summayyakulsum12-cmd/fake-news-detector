"""
Trains the TF-IDF + Logistic Regression baseline classifier that forms the
"fast, statistical" half of the hybrid pipeline.

Usage:
    python src/train_ml_model.py --data data/sample_news.csv

Swap --data for a larger corpus (e.g. the Kaggle "Fake and Real News" dataset,
ISOT dataset, or LIAR dataset) for a production-grade model. The sample CSV
shipped here is only a small demo set so the project runs end-to-end out of
the box.
"""
import argparse
import os
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import clean_text

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def main(data_path: str, test_size: float = 0.25, random_state: int = 42):
    os.makedirs(MODELS_DIR, exist_ok=True)

    df = pd.read_csv(data_path)
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("CSV must contain 'text' and 'label' columns (label: FAKE/REAL).")

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)

    X = df["clean_text"]
    y = df["label"].str.upper()

    # Small demo dataset -> skip stratified split if a class has too few samples
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_df=0.9,
        min_df=1,
        stop_words="english",
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(clf, os.path.join(MODELS_DIR, "ml_model.pkl"))
    print(f"Saved model + vectorizer to {MODELS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=os.path.join("data", "sample_news.csv"))
    parser.add_argument("--test-size", type=float, default=0.25)
    args = parser.parse_args()
    main(args.data, args.test_size)
