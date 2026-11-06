"""
evaluate.py
===========
Evaluation metrics, statistical significance testing (95% CI),
and benchmark comparisons adhering to Table 4 and Table 5 of:
Tolani et al., "Clustering-cum-regression based model and performance analysis
for early prediction of heart disease", Scientific Reports (2026).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Reference metrics published in Table 4 of the paper
PAPER_BENCHMARKS = {
    'Decision Tree Regression': {
        'TP': 32, 'FP': 8, 'FN': 8, 'TN': 20,
        'Accuracy': 0.7647, 'Recall': 0.8000, 'Specificity': 0.7143,
        'F1 Score': 0.7619, 'ROC–AUC': 0.7624,
        'CI_Lower': 0.6639, 'CI_Upper': 0.8655
    },
    'K-Nearest Neighbor': {
        'TP': 32, 'FP': 8, 'FN': 7, 'TN': 21,
        'Accuracy': 0.7794, 'Recall': 0.8205, 'Specificity': 0.7241,
        'F1 Score': 0.7742, 'ROC–AUC': 0.8896,
        'CI_Lower': 0.6809, 'CI_Upper': 0.8780
    },
    'Support Vector Machine': {
        'TP': 32, 'FP': 8, 'FN': 6, 'TN': 22,
        'Accuracy': 0.7941, 'Recall': 0.8421, 'Specificity': 0.7333,
        'F1 Score': 0.7843, 'ROC–AUC': 0.9049,
        'CI_Lower': 0.6980, 'CI_Upper': 0.8902
    },
    'Kernel SVM': {
        'TP': 32, 'FP': 8, 'FN': 7, 'TN': 21,
        'Accuracy': 0.7794, 'Recall': 0.8205, 'Specificity': 0.7241,
        'F1 Score': 0.7742, 'ROC–AUC': 0.9048,
        'CI_Lower': 0.6809, 'CI_Upper': 0.8780
    },
    'Logistic Regression': {
        'TP': 32, 'FP': 8, 'FN': 6, 'TN': 22,
        'Accuracy': 0.7941, 'Recall': 0.8421, 'Specificity': 0.7333,
        'F1 Score': 0.7843, 'ROC–AUC': 0.9019,
        'CI_Lower': 0.6980, 'CI_Upper': 0.8902
    },
    'Naïve Bayes': {
        'TP': 32, 'FP': 8, 'FN': 7, 'TN': 21,
        'Accuracy': 0.7794, 'Recall': 0.8205, 'Specificity': 0.7241,
        'F1 Score': 0.7742, 'ROC–AUC': 0.9062,
        'CI_Lower': 0.6809, 'CI_Upper': 0.8780
    },
    'Random Forest Regression': {
        'TP': 37, 'FP': 8, 'FN': 9, 'TN': 19,
        'Accuracy': 0.8235, 'Recall': 0.8043, 'Specificity': 0.7037,
        'F1 Score': 0.7989, 'ROC–AUC': 0.9167,
        'CI_Lower': 0.6702, 'CI_Upper': 0.8641
    },
    'Random Forest with K-Means Analysis': {
        'TP': 39, 'FP': 1, 'FN': 5, 'TN': 23,
        'Accuracy': 0.9100, 'Recall': 0.8864, 'Specificity': 0.9583,
        'F1 Score': 0.8977, 'ROC–AUC': 0.9155,
        'CI_Lower': 0.8443, 'CI_Upper': 0.9792
    }
}

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Computes all standard metrics matching Table 4 of the paper:
    TP, FP, FN, TN, Accuracy, Recall, Specificity, F1 Score, ROC-AUC, 95% CI.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    n = len(y_true)
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = np.nan
    else:
        auc = np.nan

    # 95% Confidence Interval for Accuracy using Wald interval
    ci_margin = 1.96 * np.sqrt((acc * (1.0 - acc)) / max(n, 1))
    ci_lower = max(0.0, acc - ci_margin)
    ci_upper = min(1.0, acc + ci_margin)

    return {
        'TP': int(tp),
        'FP': int(fp),
        'FN': int(fn),
        'TN': int(tn),
        'Accuracy': round(acc, 4),
        'Recall': round(rec, 4),
        'Specificity': round(spec, 4),
        'F1 Score': round(f1, 4),
        'ROC–AUC': round(auc, 4) if not np.isnan(auc) else None,
        'CI_Lower': round(ci_lower, 4),
        'CI_Upper': round(ci_upper, 4)
    }

def build_comparison_table(results_dict):
    """
    Constructs a comprehensive comparison DataFrame matching the paper's results
    with replication results and differences.
    """
    rows = []
    for model_name, res in results_dict.items():
        paper_res = PAPER_BENCHMARKS.get(model_name, {})
        row = {
            'Model': model_name,
            'Paper Accuracy': paper_res.get('Accuracy', 'N/A'),
            'Our Accuracy': res.get('Accuracy', 'N/A'),
            'Diff Accuracy': round(res.get('Accuracy', 0) - paper_res.get('Accuracy', 0), 4) if 'Accuracy' in paper_res else 'N/A',
            'Paper F1': paper_res.get('F1 Score', 'N/A'),
            'Our F1': res.get('F1 Score', 'N/A'),
            'Diff F1': round(res.get('F1 Score', 0) - paper_res.get('F1 Score', 0), 4) if 'F1 Score' in paper_res else 'N/A',
            'Paper Recall': paper_res.get('Recall', 'N/A'),
            'Our Recall': res.get('Recall', 'N/A'),
            'Paper Specificity': paper_res.get('Specificity', 'N/A'),
            'Our Specificity': res.get('Specificity', 'N/A'),
            'Paper ROC-AUC': paper_res.get('ROC–AUC', 'N/A'),
            'Our ROC-AUC': res.get('ROC–AUC', 'N/A')
        }
        rows.append(row)
    return pd.DataFrame(rows)
