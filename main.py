"""
main.py
=======
End-to-end replication script for:
Tolani et al., "Clustering-cum-regression based model and performance analysis
for early prediction of heart disease", Scientific Reports (2026).
DOI: 10.1038/s41598-026-40626-z

Usage:
    python main.py
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.data_loader import prepare_dataset
from src.preprocessing import (
    get_train_test_split,
    scale_features,
    compute_elbow_inertias,
    fit_kmeans_clusterer,
    add_cluster_features
)
from src.models import get_baseline_models, KMeansRandomForestHybrid
from src.evaluate import calculate_metrics, build_comparison_table, PAPER_BENCHMARKS
from src.utils import (
    plot_elbow_curve,
    plot_clustering_analysis,
    plot_confusion_matrices,
    plot_roc_curves
)

def run_experiment(random_state=42):
    print("=" * 80)
    print("REPLICATION: Clustering-cum-Regression for Early Prediction of Heart Disease")
    print("Nature Scientific Reports (2026) | DOI: 10.1038/s41598-026-40626-z")
    print("=" * 80)

    # 1. Load Data
    print("\n[Step 1] Loading and Preprocessing UCI Cleveland Heart Disease Dataset...")
    df, X, y = prepare_dataset(drop_ca_thal=True, binarize_target=True)
    print(f" -> Total Samples: {len(df)}")
    print(f" -> Features retained ({X.shape[1]}): {list(X.columns)}")
    print(f" -> Target Distribution: Healthy (0) = {(y == 0).sum()}, Diseased (1) = {(y == 1).sum()}")

    # 2. Train / Test Split (80% train, 20% test)
    print("\n[Step 2] Splitting dataset (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y, test_size=0.20, random_state=random_state)
    print(f" -> Training set size: {len(X_train)} samples")
    print(f" -> Testing set size: {len(X_test)} samples")

    # 3. Scaling (Min-Max normalization as per Section 3)
    print("\n[Step 3] Applying Min-Max Scaling [0, 1]...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='minmax')

    # 4. Exploratory Data Analysis & Clustering Visualizations
    print("\n[Step 4] Generating EDA & Clustering Analysis (Elbow Method & Scatter Plots)...")
    try:
        k_vals, inertias = compute_elbow_inertias(X_train_scaled, k_range=range(1, 11), random_state=random_state)
        plot_elbow_curve(k_vals, inertias)
        plot_clustering_analysis(df)
    except Exception as e:
        print(f" -> Visualization warning: {e}")

    # 5. Model Evaluation
    print("\n[Step 5] Training and Evaluating Machine Learning Models...")
    models = get_baseline_models(random_state=random_state)

    # Add Proposed Hybrid Models
    # Hybrid 1: Feature Augmented (Cluster labels + Centroid distances fed to RF)
    models['Hybrid (RF + K-Means Feat)'] = KMeansRandomForestHybrid(
        n_clusters=4,
        mode='feature_augmented',
        random_state=random_state
    )

    # Hybrid 2: Cluster Routing (Per-cluster RF classifier per Equation 11)
    models['Random Forest with K-Means Analysis'] = KMeansRandomForestHybrid(
        n_clusters=4,
        mode='cluster_routing',
        random_state=random_state
    )

    results = {}
    cms = []
    model_probs = {}

    for name, model in models.items():
        print(f" -> Training {name}...")
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        # Probabilities for ROC-AUC
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            except Exception:
                y_prob = None

        metrics = calculate_metrics(y_test, y_pred, y_prob)
        results[name] = metrics
        cms.append(confusion_matrix(y_test, y_pred))
        model_probs[name] = y_prob

    # 6. Generate Figures
    print("\n[Step 6] Generating Confusion Matrices and ROC Curves...")
    try:
        plot_confusion_matrices(cms, list(models.keys()))
        plot_roc_curves(y_test, model_probs)
    except Exception as e:
        print(f" -> Plotting warning: {e}")

    # 7. Comparison Table
    print("\n[Step 7] Generating Comparison Table against Paper Benchmarks...")
    comp_df = build_comparison_table(results)
    
    os.makedirs("results", exist_ok=True)
    comp_df.to_csv("results/metrics_comparison.csv", index=False)
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON TABLE (Replication vs Published Paper)")
    print("=" * 80)
    print(comp_df.to_string(index=False))

    # Print Confidence Intervals
    print("\n" + "=" * 80)
    print("95% CONFIDENCE INTERVALS FOR ACCURACY")
    print("=" * 80)
    ci_rows = []
    for m, met in results.items():
        ci_rows.append({
            'Model': m,
            'Accuracy': met['Accuracy'],
            '95% CI Lower': met['CI_Lower'],
            '95% CI Upper': met['CI_Upper']
        })
    print(pd.DataFrame(ci_rows).to_string(index=False))

    return results, comp_df

if __name__ == "__main__":
    run_experiment()

