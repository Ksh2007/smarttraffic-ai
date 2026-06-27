# =============================================================
# SmartTraffic AI — Phase 3: Machine Learning (DT + KNN)
# Run: python phase3_ml.py
# =============================================================

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

FEATURES = ["vehicle_speed", "vehicle_count", "road_enc", "weather_enc"]


# ------------------------------------------------------------------
# Encoding
# ------------------------------------------------------------------
def encode_features(df_sensor: pd.DataFrame):
    """
    Label-encode categorical columns and return the encoded DataFrame
    plus the three fitted encoders.
    """
    df = df_sensor.copy()

    le_road    = LabelEncoder()
    le_weather = LabelEncoder()
    le_label   = LabelEncoder()

    df["road_enc"]    = le_road.fit_transform(df["road"])
    df["weather_enc"] = le_weather.fit_transform(df["weather"])
    df["label_enc"]   = le_label.fit_transform(df["congestion_level"])

    print("Congestion class mapping:",
          dict(zip(le_label.classes_, le_label.transform(le_label.classes_))))

    return df, le_road, le_weather, le_label


def prepare_splits(df_encoded: pd.DataFrame, test_size: float = 0.2):
    """Return X_train, X_test, y_train, y_test."""
    X = df_encoded[FEATURES]
    y = df_encoded["label_enc"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------
# Decision Tree
# ------------------------------------------------------------------
def train_decision_tree(X_train, y_train, max_depth: int = 5):
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, label_names, model_name: str):
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_names))
    return y_pred, acc


# ------------------------------------------------------------------
# KNN
# ------------------------------------------------------------------
def train_knn(X_train, y_train, n_neighbors: int = 5):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train_sc, y_train)
    return model, scaler


# ------------------------------------------------------------------
# Feature importance (Decision Tree only)
# ------------------------------------------------------------------
def print_feature_importance(dt_model, feature_names):
    importance_df = pd.DataFrame({
        "feature":    feature_names,
        "importance": dt_model.feature_importances_.round(4),
    }).sort_values("importance", ascending=False)
    print("\nFeature importances (Decision Tree):")
    print(importance_df.to_string(index=False))


# ------------------------------------------------------------------
# Save artifacts
# ------------------------------------------------------------------
def save_ml_artifacts(dt_model, knn_model, scaler, le_road, le_weather, le_label):
    os.makedirs("models", exist_ok=True)
    artifacts = {
        "models/dt_model.pkl":     dt_model,
        "models/knn_model.pkl":    knn_model,
        "models/scaler.pkl":       scaler,
        "models/le_road.pkl":      le_road,
        "models/le_weather.pkl":   le_weather,
        "models/le_label.pkl":     le_label,
    }
    for path, obj in artifacts.items():
        with open(path, "wb") as f:
            pickle.dump(obj, f)
    print("\n✅  ML artifacts saved to /models/")


def load_ml_artifacts():
    files = ["dt_model", "knn_model", "scaler", "le_road", "le_weather", "le_label"]
    loaded = {}
    for name in files:
        with open(f"models/{name}.pkl", "rb") as f:
            loaded[name] = pickle.load(f)
    return loaded


# ------------------------------------------------------------------
# Single-sample prediction helper (used by Phase 4)
# ------------------------------------------------------------------
def predict_congestion(
    road: str,
    speed: int,
    count: int,
    weather: str,
    dt_model,
    le_road,
    le_weather,
    le_label,
) -> str:
    road_enc    = le_road.transform([road])[0] if road in le_road.classes_ else 0
    weather_enc = le_weather.transform([weather])[0] if weather in le_weather.classes_ else 0
    features    = pd.DataFrame(
        [[speed, count, road_enc, weather_enc]], columns=FEATURES
    )
    encoded = dt_model.predict(features)[0]
    return le_label.inverse_transform([encoded])[0]


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    print("Loading sensor dataset …")
    df_sensor = pd.read_csv("data/sensor_data.csv")

    print("\nEncoding features …")
    df_encoded, le_road, le_weather, le_label = encode_features(df_sensor)

    print("\nPreparing train/test splits …")
    X_train, X_test, y_train, y_test = prepare_splits(df_encoded)

    label_names = list(le_label.classes_)

    # --- Decision Tree ---
    print("\nTraining Decision Tree …")
    dt_model = train_decision_tree(X_train, y_train, max_depth=5)
    y_pred_dt, acc_dt = evaluate_model(dt_model, X_test, y_test, label_names, "Decision Tree")
    print_feature_importance(dt_model, FEATURES)

    # --- KNN ---
    print("\nTraining KNN (k=5) …")
    knn_model, scaler = train_knn(X_train, y_train, n_neighbors=5)
    X_test_sc = scaler.transform(X_test)
    y_pred_knn, acc_knn = evaluate_model(
        knn_model, X_test_sc, y_test, label_names, "KNN (k=5)"
    )

    # --- Comparison ---
    print("\n" + "="*40)
    print("  Model comparison")
    print("="*40)
    print(f"  Decision Tree accuracy : {acc_dt:.4f}")
    print(f"  KNN accuracy           : {acc_knn:.4f}")
    winner = "Decision Tree" if acc_dt >= acc_knn else "KNN"
    print(f"  Best model             : {winner}")

    # --- Demo prediction ---
    sample = predict_congestion(
        road="Ring Road", speed=8, count=500, weather="Rainy",
        dt_model=dt_model, le_road=le_road, le_weather=le_weather, le_label=le_label,
    )
    print(f"\nDemo prediction  →  Ring Road, 8 km/h, 500 vehicles, Rainy  →  '{sample}'")

    save_ml_artifacts(dt_model, knn_model, scaler, le_road, le_weather, le_label)
    print("\nPhase 3 complete.")


if __name__ == "__main__":
    main()
