"""
models.py
=========
Machine Learning models implemented as specified in:
Tolani et al., "Clustering-cum-regression based model and performance analysis
for early prediction of heart disease", Scientific Reports (2026).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans

def get_baseline_models(random_state=42):
    """
    Returns the exact baseline models and hyperparameter configurations
    from Table 3 and Table 4 of the paper.
    """
    models = {
        'Decision Tree Regression': DecisionTreeClassifier(
            max_depth=5,
            random_state=random_state
        ),
        'K-Nearest Neighbor': KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean',
            algorithm='auto'
        ),
        'Support Vector Machine': SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced',
            probability=True,
            random_state=random_state
        ),
        'Kernel SVM': SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=random_state
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        ),
        'Naïve Bayes': GaussianNB(),
        'Random Forest Regression': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=random_state
        )
    }
    return models

class KMeansRandomForestHybrid(BaseEstimator, ClassifierMixin):
    """
    Proposed Hybrid Model combining K-Means Clustering with Random Forest.
    
    Supports two architectural interpretations described in the paper:
    1. 'feature_augmented': K-Means is used as unsupervised pre-processing to enrich 
       feature representation (cluster IDs and centroid distances) before training Random Forest.
    2. 'cluster_routing': (Equation 11) Points are assigned to nearest cluster centroid k*,
       and classified by cluster-specific Random Forest model F_k*.
    """
    def __init__(self, n_clusters=4, n_estimators=200, max_depth=10, 
                 mode='feature_augmented', random_state=42):
        self.n_clusters = n_clusters
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.mode = mode
        self.random_state = random_state
        self.kmeans = None
        self.global_rf = None
        self.cluster_models = {}

    def _augment(self, X):
        cluster_labels = self.kmeans.predict(X).reshape(-1, 1)
        distances = self.kmeans.transform(X)
        return np.hstack([X, cluster_labels, distances])

    def fit(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y)

        # 1. Fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans.fit(X_arr)

        if self.mode == 'feature_augmented':
            X_aug = self._augment(X_arr)
            self.global_rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_state
            )
            self.global_rf.fit(X_aug, y_arr)

        elif self.mode == 'cluster_routing':
            cluster_assignments = self.kmeans.labels_
            self.global_rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.global_rf.fit(X_arr, y_arr)

            for k in range(self.n_clusters):
                idx = np.where(cluster_assignments == k)[0]
                if len(idx) > 5 and len(np.unique(y_arr[idx])) > 1:
                    rf_k = RandomForestClassifier(
                        n_estimators=self.n_estimators,
                        max_depth=self.max_depth,
                        random_state=self.random_state
                    )
                    rf_k.fit(X_arr[idx], y_arr[idx])
                    self.cluster_models[k] = rf_k
                else:
                    # Fallback to global model if cluster too small or homogenous
                    self.cluster_models[k] = self.global_rf

        return self

    def predict(self, X):
        X_arr = np.array(X)
        if self.mode == 'feature_augmented':
            X_aug = self._augment(X_arr)
            return self.global_rf.predict(X_aug)

        elif self.mode == 'cluster_routing':
            clusters = self.kmeans.predict(X_arr)
            preds = np.zeros(len(X_arr), dtype=int)
            for k in range(self.n_clusters):
                idx = np.where(clusters == k)[0]
                if len(idx) > 0:
                    model = self.cluster_models.get(k, self.global_rf)
                    preds[idx] = model.predict(X_arr[idx])
            return preds

    def predict_proba(self, X):
        X_arr = np.array(X)
        if self.mode == 'feature_augmented':
            X_aug = self._augment(X_arr)
            return self.global_rf.predict_proba(X_aug)

        elif self.mode == 'cluster_routing':
            clusters = self.kmeans.predict(X_arr)
            probs = np.zeros((len(X_arr), 2))
            for k in range(self.n_clusters):
                idx = np.where(clusters == k)[0]
                if len(idx) > 0:
                    model = self.cluster_models.get(k, self.global_rf)
                    p = model.predict_proba(X_arr[idx])
                    if p.shape[1] == 1:
                        # Only one class present
                        cls = model.classes_[0]
                        probs[idx, cls] = 1.0
                    else:
                        probs[idx] = p
            return probs

