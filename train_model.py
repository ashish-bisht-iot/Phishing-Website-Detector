import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from feature_extractor import FEATURE_VECTOR_KEYS

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'phishing_model.pkl')
DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'phishing_dataset.csv')

os.makedirs(MODEL_DIR, exist_ok=True)


def load_dataset(path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the UCI Phishing Dataset.
    Expects CSV with numeric feature columns + 'Result' column (1=legit, -1=phishing).
    Falls back to synthetic data if file not found.
    """
    if os.path.exists(path):
        print(f"Loading dataset from {path}")
        df = pd.read_csv(path)

        if 'Result' in df.columns:
            y_raw = df['Result'].values
            # UCI uses 1=legit, -1=phishing → convert to 0=safe, 1=phishing
            y = np.where(y_raw == -1, 1, 0)
            X = df.drop(columns=['Result']).values
        elif 'label' in df.columns:
            y = df['label'].values
            X = df.drop(columns=['label']).values
        else:
            raise ValueError("Dataset must have 'Result' or 'label' column")

        print(f"  Loaded {len(y)} samples | phishing={y.sum()} | legit={len(y)-y.sum()}")
        return X, y

    else:
        print("Dataset not found — generating synthetic training data...")
        return _generate_synthetic_data()


def _generate_synthetic_data(n=5000) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate realistic synthetic feature vectors for demonstration.
    In production: download the UCI Phishing Dataset from
    https://archive.ics.uci.edu/dataset/327/phishing+websites
    """
    rng = np.random.default_rng(42)
    n_features = len(FEATURE_VECTOR_KEYS)
    half = n // 2

    # Legit URLs: long URLs less common, HTTPS common, few keywords
    legit = rng.normal(
        loc=[55, 15, 12, 0.9, 0, 5, 0.2, 0.1, 4, 0.2, 1, 0, 0, 0, 3, 0.05,
             1, 0, 0.2, 2, 0.3, 0.1, 0, 2, 0, 0, 0.7, 400, 1],
        scale=[20, 8, 10, 0.1, 0.05, 2, 0.4, 0.3, 2, 0.4, 1, 0.05, 0.05, 0.05,
               3, 0.05, 1, 0.1, 0.4, 3, 0.4, 0.3, 0.1, 1, 0.05, 0.1, 0.2, 300, 0.05],
        size=(half, n_features)
    )

    # Phishing URLs: longer, HTTP, IPs, suspicious TLDs, many keywords
    phish = rng.normal(
        loc=[90, 30, 25, 0.2, 0.3, 8, 2, 0.5, 7, 1, 3, 0.2, 0.1, 0.3, 10, 0.12,
             5, 0.6, 2, 8, 0, 3, 0.5, 5, 0.1, 0.2, 0.8, 30, 0.6],
        scale=[30, 12, 15, 0.2, 0.3, 3, 1, 0.4, 3, 0.8, 2, 0.3, 0.2, 0.4,
               5, 0.06, 3, 0.3, 1, 6, 0.1, 2, 0.4, 2, 0.2, 0.3, 0.2, 40, 0.3],
        size=(half, n_features)
    )

    X = np.vstack([legit, phish])
    y = np.array([0] * half + [1] * half)

    # Clip to realistic bounds
    X = np.clip(X, 0, None)
    X[:, 3] = np.clip(X[:, 3], 0, 1).round()   # has_https
    X[:, 4] = np.clip(X[:, 4], 0, 1).round()   # has_ip
    X[:, 11] = np.clip(X[:, 11], 0, 1).round()  # at_symbol
    X[:, 17] = np.clip(X[:, 17], 0, 1).round()  # suspicious_tld
    X[:, 20] = np.clip(X[:, 20], 0, 1).round()  # is_trusted_domain

    idx = rng.permutation(n)
    return X[idx].astype(np.float32), y[idx]


def build_model() -> Pipeline:
    """Build a voting ensemble of Random Forest + Gradient Boosting."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
    )
    gb = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=6,
        subsample=0.85,
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft',
        weights=[2, 1],
    )
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', ensemble),
    ])
    return pipeline


def train():
    print("=" * 60)
    print("Phishing Detector — Model Training")
    print("=" * 60)

    X, y = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

    model = build_model()

    print("\nTraining ensemble model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legit', 'Phishing']))

    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    auc = roc_auc_score(y_test, y_prob)
    print(f"\nROC-AUC Score: {auc:.4f}")

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='f1', n_jobs=-1)
    print(f"5-fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    metadata = {
        'model': model,
        'feature_keys': FEATURE_VECTOR_KEYS,
        'auc': auc,
        'cv_f1_mean': cv_scores.mean(),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"\nModel saved to {MODEL_PATH}")
    print("=" * 60)


if __name__ == '__main__':
    train()
