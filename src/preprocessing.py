"""
preprocessing.py
================
Preprocessing pipeline adhering to:
Tolani et al., "Clustering-cum-regression based model and performance analysis
for early prediction of heart disease", Scientific Reports (2026).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans

def get_train_test_split(X, y, test_size=0.20, random_state=42):
    """
    Performs stratified train/test split (80% train, 20% test)
    to maintain balanced class proportions across splits.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test

def scale_features(X_train, X_test, method='minmax'):
    """
    Applies Min-Max scaling to [0, 1] range (or Z-score Standardization)
    fit strictly on X_train to prevent data leakage.
    """
    if method == 'minmax':
        scaler = MinMaxScaler(feature_range=(0, 1))
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    cols = X_train.columns if hasattr(X_train, 'columns') else None
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if cols is not None:
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=cols, index=X_train.index)
        X_test_scaled = pd.DataFrame(X_test_scaled, columns=cols, index=X_test.index)

    return X_train_scaled, X_test_scaled, scaler

def compute_elbow_inertias(X, k_range=range(1, 11), random_state=42):
    """
    Calculates within-cluster sum of squares (inertia) across varying K
    to find the optimal number of clusters via the elbow method.
    """
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)
    return list(k_range), inertias

def fit_kmeans_clusterer(X_train, n_clusters=4, random_state=42):
    """
    Fits K-Means clustering model on training features.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train)
    return kmeans

def add_cluster_features(X, kmeans):
    """
    Augments the feature matrix with cluster assignments and
    distances to each cluster centroid, enriching input context for the classifier.
    """
    cluster_labels = kmeans.predict(X)
    cluster_distances = kmeans.transform(X)  # Euclidean distances to all centroids

    X_augmented = X.copy()
    if isinstance(X_augmented, pd.DataFrame):
        X_augmented['cluster_label'] = cluster_labels
        for c in range(cluster_distances.shape[1]):
            X_augmented[f'dist_to_cluster_{c}'] = cluster_distances[:, c]
    else:
        dist_cols = cluster_distances
        labels_col = cluster_labels.reshape(-1, 1)
        X_augmented = np.hstack([X_augmented, labels_col, dist_cols])

    return X_augmented

