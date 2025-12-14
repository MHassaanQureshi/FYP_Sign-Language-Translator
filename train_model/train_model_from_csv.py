# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------- Load data ----------
def load_and_preprocess_data(filename):
    df = pd.read_csv(filename)
    if df.isnull().any().any():
        print("Warning: Missing data found. Dropping rows...")
        df = df.dropna()
    X = df.iloc[:, :-1].values  # 126 columns
    y = df.iloc[:, -1].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return X, y_enc, le

# ---------- Feature extraction (must match prediction) ----------
def extract_combined_features(row):
    """
    row = (126,) → [left(63) + right(63)]
    returns 144 features (63 rel left + 63 rel right + 5 dist left + 5 dist right + 4 angles left + 4 angles right)
    """
    row = np.array(row)
    h1 = row[:63].reshape(21, 3)
    h2 = row[63:].reshape(21, 3)

    def finger_angles(hand):
        fingertips = [4, 8, 12, 16, 20]
        vecs = [(hand[i] - hand[0]) for i in fingertips]
        angs = []
        for i in range(len(vecs) - 1):
            cos_val = np.dot(vecs[i], vecs[i+1]) / (np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[i+1]) + 1e-9)
            angs.append(np.arccos(np.clip(cos_val, -1, 1)))
        return angs

    # left features
    h1_rel = (h1 - h1[0]).flatten()
    h1_dist = [np.linalg.norm(h1[i] - h1[0]) for i in [4,8,12,16,20]]
    h1_angles = finger_angles(h1)

    # check if second hand exists (all zeros)
    if np.all(h2 == 0):
        # pad zeros
        h2_rel = np.zeros_like(h1_rel)
        h2_dist = [0]*5
        h2_angles = [0]*4
    else:
        h2_rel = (h2 - h2[0]).flatten()
        h2_dist = [np.linalg.norm(h2[i] - h2[0]) for i in [4,8,12,16,20]]
        h2_angles = finger_angles(h2)

    features = np.concatenate([h1_rel, h2_rel, np.array(h1_dist), np.array(h2_dist), np.array(h1_angles), np.array(h2_angles)])
    return features

# ---------- Training ----------
def train_model(X, y):
    print("Extracting features...")
    X_feat = np.array([extract_combined_features(x) for x in X])

    X_train, X_test, y_train, y_test = train_test_split(X_feat, y, test_size=0.2, random_state=42, stratify=y)

    pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=300, max_depth=18, class_weight='balanced', random_state=42))
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("\nModel evaluation:")
    print(classification_report(y_test, preds))

    return pipeline

def main():
    X, y, le = load_and_preprocess_data("double_hand_dataset.csv")
    model = train_model(X, y)
    joblib.dump(model, "gesture_recognition_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("\nSaved model and label encoder.")

if __name__ == "__main__":
    main()
