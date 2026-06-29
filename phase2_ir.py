# =============================================================
# SmartTraffic AI — Phase 2: NLP Preprocessing + IR (TF-IDF)
# Run: python phase2_ir.py
# =============================================================

import re
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

nltk.download("stopwords", quiet=True)
nltk.download("punkt",     quiet=True)

STOP_WORDS = set(stopwords.words("english"))
STEMMER    = PorterStemmer()


# ------------------------------------------------------------------
# Text preprocessing
# ------------------------------------------------------------------
def preprocess_text(text: str) -> str:
    """Lowercase → remove punctuation/digits → stop-word removal → stemming."""
    text   = text.lower()
    text   = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [STEMMER.stem(t) for t in tokens]
    return " ".join(tokens)


def preprocess_dataframes(
    df_accident:   pd.DataFrame,
    df_police:     pd.DataFrame,
    df_complaints: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply preprocess_text to all text columns and return updated frames."""
    df_accident   = df_accident.copy()
    df_police     = df_police.copy()
    df_complaints = df_complaints.copy()

    df_accident["report_clean"]         = df_accident["report"].apply(preprocess_text)
    df_police["police_report_clean"]    = df_police["police_report"].apply(preprocess_text)
    df_complaints["complaint_clean"]    = df_complaints["complaint"].apply(preprocess_text)

    return df_accident, df_police, df_complaints


# ------------------------------------------------------------------
# TF-IDF corpus builder
# ------------------------------------------------------------------
def build_corpus(
    df_accident:   pd.DataFrame,
    df_police:     pd.DataFrame,
    df_complaints: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all cleaned text columns into one searchable corpus."""
    return pd.concat([
        df_accident[["report_clean"]].rename(
            columns={"report_clean": "text"}).assign(source="accident"),
        df_police[["police_report_clean"]].rename(
            columns={"police_report_clean": "text"}).assign(source="police"),
        df_complaints[["complaint_clean"]].rename(
            columns={"complaint_clean": "text"}).assign(source="complaint"),
    ], ignore_index=True)


def fit_tfidf(corpus: pd.DataFrame, max_features: int = 500):
    """Fit TF-IDF vectorizer and return (vectorizer, tfidf_matrix)."""
    vectorizer   = TfidfVectorizer(max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(corpus["text"])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    return vectorizer, tfidf_matrix


# ------------------------------------------------------------------
# Retrieval function
# ------------------------------------------------------------------
def retrieve_documents(
    query:        str,
    corpus:       pd.DataFrame,
    vectorizer:   TfidfVectorizer,
    tfidf_matrix,
    top_k:        int = 5,
) -> pd.DataFrame:
    """Return the top-k most relevant documents for a query."""
    query_clean = preprocess_text(query)
    query_vec   = vectorizer.transform([query_clean])
    scores      = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = scores.argsort()[::-1][:top_k]
    results     = corpus.iloc[top_indices].copy()
    results["score"] = scores[top_indices].round(4)
    return results[["source", "text", "score"]]


# ------------------------------------------------------------------
# Save / load IR artifacts
# ------------------------------------------------------------------
def save_ir_artifacts(vectorizer, tfidf_matrix, corpus: pd.DataFrame):
    os.makedirs("models", exist_ok=True)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("models/tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    corpus.to_csv("models/corpus.csv", index=False)
    print("✅  IR artifacts saved to /models/")


def load_ir_artifacts():
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    corpus = pd.read_csv("models/corpus.csv")
    return vectorizer, tfidf_matrix, corpus


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading datasets …")
    df_accident   = pd.read_csv("data/accident_data.csv")
    df_police     = pd.read_csv("data/police_data.csv")
    df_complaints = pd.read_csv("data/complaints.csv")

    print("Preprocessing text …")
    df_accident, df_police, df_complaints = preprocess_dataframes(
        df_accident, df_police, df_complaints
    )

    # Show a sample before/after
    print("\nSample original :", df_accident["report"].iloc[0])
    print("Sample cleaned  :", df_accident["report_clean"].iloc[0])

    # Build TF-IDF
    print("\nBuilding TF-IDF …")
    corpus = build_corpus(df_accident, df_police, df_complaints)
    vectorizer, tfidf_matrix = fit_tfidf(corpus)

    # Save
    save_ir_artifacts(vectorizer, tfidf_matrix, corpus)

    # Save preprocessed DataFrames for Phase 3
    df_accident.to_csv("data/accident_data_clean.csv",  index=False)
    df_police.to_csv("data/police_data_clean.csv",      index=False)
    df_complaints.to_csv("data/complaints_clean.csv",   index=False)

    # Demo queries
    test_queries = [
        "accidents near Ring Road",
        "signal failure congestion",
        "truck blocking lane",
    ]
    for q in test_queries:
        print(f"\n--- Query: '{q}' ---")
        results = retrieve_documents(q, corpus, vectorizer, tfidf_matrix, top_k=3)
        print(results.to_string(index=False))

    print("\nPhase 2 complete.")


if __name__ == "__main__":
    main()
