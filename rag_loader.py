
import pickle
import pandas as pd
def load_all_artifacts():
    """Load IR and ML artifacts from /models/."""
    def _load(name):
        with open(f"models/{name}.pkl", "rb") as f:
            return pickle.load(f)

    return {
        "vectorizer":   _load("tfidf_vectorizer"),
        "tfidf_matrix": _load("tfidf_matrix"),
        "corpus":       pd.read_csv("models/corpus.csv"),
        "dt_model":     _load("dt_model"),
        "le_road":      _load("le_road"),
        "le_weather":   _load("le_weather"),
        "le_label":     _load("le_label"),
    }