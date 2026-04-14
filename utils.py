import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data(path="data.xlsx"):
    df = pd.read_excel(path)
    df = df.drop(columns=["Unnamed: 0"])
    const_cols = [c for c in df.columns if df[c].nunique() <= 1]
    df = df.drop(columns=const_cols)
    df = df.fillna(df.median(numeric_only=True))
    return df


def get_features_targets(df):
    targets = ["IC50, mM", "CC50, mM", "SI"]
    feature_cols = [c for c in df.columns if c not in targets]
    return df[feature_cols], df[targets]


def get_regression_split(df, target_col, log_transform=True):
    X, targets = get_features_targets(df)
    y = targets[target_col]
    if log_transform:
        y = np.log1p(y)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def get_classification_split(df, target_col, threshold=None):
    X, targets = get_features_targets(df)
    y_raw = targets[target_col]
    if threshold is None:
        threshold = y_raw.median()
    y = (y_raw > threshold).astype(int)
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
