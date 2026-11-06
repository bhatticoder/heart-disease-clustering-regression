"""
utils.py
========
Visualization, logging, and reporting utilities for:
Tolani et al., "Clustering-cum-regression based model and performance analysis
for early prediction of heart disease", Scientific Reports (2026).
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def set_style():
    """Sets a clean, publication-ready style for plots."""
    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

def plot_elbow_curve(k_range, inertias, output_path="results/figures/elbow_plot.png"):
    """
    Plots Within-Cluster Sum of Squares (Inertia) vs K
    Replicating Figure 1 of the paper.
    """
    set_style()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_range, inertias, marker='o', linewidth=2, color='#1f77b4', markersize=7)
    ax.set_title("Elbow Method for Optimal Clusters (K-Means)", fontweight='bold')
    ax.set_xlabel("Number of Clusters (K)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_xticks(k_range)
    
    # Highlight elbow point (typically K=4)
    if len(k_range) >= 4:
        elbow_k = 4
        elbow_idx = list(k_range).index(elbow_k)
        ax.plot(elbow_k, inertias[elbow_idx], marker='o', markersize=12, color='red', fillstyle='none', mew=2.5)
        ax.annotate(f"Optimal K = {elbow_k}", xy=(elbow_k, inertias[elbow_idx]),
                    xytext=(elbow_k + 0.8, inertias[elbow_idx] * 1.15),
                    arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=8),
                    fontweight='bold', color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved elbow plot: {output_path}")

def plot_clustering_analysis(df, output_prefix="results/figures/clustering_"):
    """
    Replicates Figure 3 (BP vs Age) and Figure 5 (Cholesterol vs Age) from the paper.
    """
    set_style()
    from sklearn.cluster import KMeans

    # 1. Blood Pressure vs Age
    X_bp = df[['age', 'trestbps']].dropna()
    km_bp = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_bp)
    X_bp['cluster'] = km_bp.labels_

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_bp['age'], X_bp['trestbps'], c=X_bp['cluster'], cmap='viridis', s=50, alpha=0.85, edgecolors='w')
    ax.scatter(km_bp.cluster_centers_[:, 0], km_bp.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    ax.set_title("Clustering of Resting Blood Pressure vs Age (K=3)", fontweight='bold')
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Resting Blood Pressure (mm Hg)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}bp_vs_age.png", dpi=300)
    plt.close()

    # 2. Cholesterol vs Age
    X_chol = df[['age', 'chol']].dropna()
    km_chol = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_chol)
    X_chol['cluster'] = km_chol.labels_

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(X_chol['age'], X_chol['chol'], c=X_chol['cluster'], cmap='plasma', s=50, alpha=0.85, edgecolors='w')
    ax.scatter(km_chol.cluster_centers_[:, 0], km_chol.cluster_centers_[:, 1], s=200, c='red', marker='X', label='Centroids')
    ax.set_title("Clustering of Serum Cholesterol vs Age (K=5)", fontweight='bold')
    ax.set_xlabel("Age (Years)")
    ax.set_ylabel("Serum Cholesterol (mg/dl)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_prefix}chol_vs_age.png", dpi=300)
    plt.close()
    print("Saved clustering analysis figures.")

def plot_confusion_matrices(cms, model_names, output_path="results/figures/confusion_matrices.png"):
    """
    Plots 2x4 grid of confusion matrices for all evaluated models.
    """
    set_style()
    n_models = len(model_names)
    cols = 4
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten()

    for idx, (name, cm) in enumerate(zip(model_names, cms)):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                    xticklabels=['Healthy (0)', 'Disease (1)'],
                    yticklabels=['Healthy (0)', 'Disease (1)'])
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    for extra in range(idx + 1, len(axes)):
        axes[extra].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices: {output_path}")

def plot_roc_curves(y_true, model_probs, output_path="results/figures/roc_curves.png"):
    """
    Plots overlaid ROC curves for all models.
    """
    from sklearn.metrics import roc_curve, auc
    set_style()
    fig, ax = plt.subplots(figsize=(9, 7))

    for name, probs in model_probs.items():
        if probs is not None:
            fpr, tpr, _ = roc_curve(y_true, probs)
            score = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {score:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guess')
    ax.set_title("ROC Curves Comparison Across Prediction Models", fontweight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved ROC curves: {output_path}")
