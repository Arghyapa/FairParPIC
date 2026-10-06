"""Loading and cleaning of the three real datasets of the paper.

Recipe (Section 6): drop columns with more than 40% missing values, keep the
numeric non-sensitive columns, median-impute, standardize, and draw a
subsample stratified by the sensitive attribute.
"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# name -> (file in data/, sensitive column, valid categories, separator)
DATASETS = {
    "diabetes": ("diabetic_data.csv", "gender", ("Male", "Female"), ","),
    "census":   ("uci_census.csv",    "gender", ("Male", "Female"), ","),
    "bank":     ("bank-full.csv",     "marital", ("married", "single", "divorced"), ";"),
}


def clean_and_filter_data(df, sensitive_col, valid_categories, missing_threshold=0.4):
    df = df.replace("?", np.nan)
    df = df.replace("unknown", np.nan)
    df = df[df[sensitive_col].isin(valid_categories)]

    missing_fraction = df.isnull().mean()
    df = df.drop(columns=missing_fraction[missing_fraction > missing_threshold].index)

    numeric_cols = df.select_dtypes(exclude=["object"]).columns
    df = df[list(numeric_cols) + [sensitive_col]].copy()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def prepare_numeric_features(df, sensitive_col):
    sensitive = df[sensitive_col].astype(str)
    groups, group_ids = np.unique(sensitive, return_inverse=True)
    X = StandardScaler().fit_transform(df.drop(columns=[sensitive_col]).values)
    return X, group_ids, groups


def load_dataset(name_or_path, sensitive_col=None, valid_categories=None, sep=",",
                 n_sample=3000, seed=42, data_dir="data"):
    """Load one of the paper's datasets by name ("diabetes", "census", "bank")
    or any CSV by path. Returns (X, group_ids, group_names)."""
    if name_or_path in DATASETS:
        fname, sensitive_col, valid_categories, sep = DATASETS[name_or_path]
        path = os.path.join(data_dir, fname)
    else:
        path = name_or_path
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; see data/README.md for download links.")
    df = clean_and_filter_data(pd.read_csv(path, sep=sep), sensitive_col, valid_categories)
    if n_sample is not None and n_sample < len(df):
        parts = []
        for _, g in df.groupby(sensitive_col):
            n_g = min(len(g), int(round(n_sample * len(g) / len(df))))
            parts.append(g.sample(n=n_g, random_state=seed))
        df = pd.concat(parts, axis=0).reset_index(drop=True)
    return prepare_numeric_features(df, sensitive_col)
