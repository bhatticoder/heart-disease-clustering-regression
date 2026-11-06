"""
data_loader.py
==============
Loads and prepares the UCI Cleveland Heart Disease dataset as described in:
Tolani et al., "Clustering-cum-regression based model and performance analysis
for early prediction of heart disease", Scientific Reports (2026).
"""

import os
import pandas as pd
import numpy as np

# Standard 14 attribute names in UCI Heart Disease (Cleveland)
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope',
    'ca', 'thal', 'num'
]

# 12 parameters retained by the paper (ca and thal excluded due to dependency and patient-specificity)
RETAINED_FEATURES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope'
]
TARGET_COL = 'num'

def load_raw_dataset(filepath="data/raw/processed.cleveland.data"):
    """
    Loads raw Cleveland dataset file, replacing '?' with NaN.
    """
    if not os.path.exists(filepath):
        # Fallback to current directory if running from different root
        if os.path.exists("processed.cleveland.data"):
            filepath = "processed.cleveland.data"
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath}")

    df = pd.read_csv(filepath, header=None, names=FEATURE_NAMES, na_values='?')
    return df

def prepare_dataset(filepath="data/raw/processed.cleveland.data", drop_ca_thal=True, binarize_target=True):
    """
    Loads dataset, handles missing values, filters features as per paper,
    and binarizes the target outcome.
    
    Returns:
        df_clean (pd.DataFrame): Processed dataframe
        X (pd.DataFrame): Input features
        y (pd.Series): Binary target labels (0 = Healthy, 1 = Heart Disease)
    """
    df = load_raw_dataset(filepath)

    # Impute missing values as documented in Section 3 of paper:
    # "Missing values are handled through imputation, i.e., the mean or median
    # for continuous attributes based on distribution characteristics...
    # and mode-based imputation for categorical variables."
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64'] and df[col].nunique() > 5:
                # Continuous: check skewness
                if abs(df[col].skew()) > 0.5:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())
            else:
                # Categorical / discrete: mode
                df[col] = df[col].fillna(df[col].mode()[0])

    # Binarize target variable: 0 = Healthy, 1+ = Heart Disease
    if binarize_target:
        df['target'] = (df['num'] > 0).astype(int)
    else:
        df['target'] = df['num']

    if drop_ca_thal:
        # Paper explicitly excludes Vessels Fluoro (ca) and Thallium (thal)
        feature_cols = RETAINED_FEATURES
    else:
        feature_cols = [c for c in FEATURE_NAMES if c != 'num']

    X = df[feature_cols].copy()
    y = df['target'].copy()

    # Save cleaned versions
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/heart_disease_clean.csv", index=False)
    df[feature_cols + ['target']].to_csv("data/processed/heart_disease_12features.csv", index=False)

    return df, X, y

if __name__ == "__main__":
    df, X, y = prepare_dataset()
    print(f"Loaded dataset: {df.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {y.value_counts().to_dict()} (0: Healthy, 1: Diseased)")

