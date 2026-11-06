# Machine Learning - Lab (CL3001)
## Semester Project: Paper Approval Form
**Instructor:** Esha Iftikhar  
**Institution:** National University of Computer & Emerging Sciences (FAST-NUCES), Chiniot-Faisalabad Campus  
**Academic Term:** Spring 2026  

---

### 1. Student Information
* **Course Title:** Machine Learning - Lab (CL3001)
* **Project Type:** Research Paper Reproduction & Comparative Performance Analysis
* **Submission Date:** November 2026

---

### 2. Selected Research Paper Details
* **Paper Title:** *Clustering-cum-regression based model and performance analysis for early prediction of heart disease*
* **Authors:** Manoj Tolani, Yazeed AlZahrani, Gaurav Suman, Arun Balodi, Ambar Bajpai, Pankaj Kumar
* **Journal / Venue:** *Scientific Reports* (Nature Portfolio), Volume 16, Article Number 40626
* **Publication Date:** February 18, 2026
* **Digital Object Identifier (DOI):** [10.1038/s41598-026-40626-z](https://doi.org/10.1038/s41598-026-40626-z)
* **Paper URL:** [https://www.nature.com/articles/s41598-026-40626-z](https://www.nature.com/articles/s41598-026-40626-z)

---

### 3. Public Dataset Information
* **Dataset Name:** UCI Heart Disease Dataset (Cleveland Database)
* **Hosting Platform:** UCI Machine Learning Repository (Center for Machine Learning and Intelligent Systems)
* **Dataset URL:** [https://archive.ics.uci.edu/ml/datasets/Heart+Disease](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
* **Data Characteristics:** 303 patient records, 14 clinical features (12 selected attributes utilized in experimental pipeline excluding patient-dependent variables `ca` and `thal`), binary target classification (0 = Healthy, 1 = Heart Disease).

---

### 4. Justification & Alignment with Course Objectives
1. **Direct Alignment with Course Topics (#CLO5):**
   The paper integrates core machine learning concepts directly taught in CL3001, including:
   * **Unsupervised Learning:** K-Means clustering, Within-Cluster Sum of Squares (Inertia), and the Elbow method.
   * **Supervised Learning & Regression/Classification:** Logistic Regression, Support Vector Machines (Linear and RBF Kernel), Decision Trees, K-Nearest Neighbors (KNN), Naïve Bayes, and Random Forest.
   * **Hybrid Modeling:** Combining unsupervised clustering representations with ensemble classifiers to capture non-linear clinical risk factors.
   * **Rigorous Evaluation:** Confusion matrix breakdown (TP, FP, FN, TN), Accuracy, Recall (Sensitivity), Specificity, F1-Score, ROC-AUC, and 95% Confidence Intervals.

2. **High Real-World Impact (Healthcare & WBAN):**
   Cardiovascular diseases (CVDs) remain the leading cause of mortality worldwide. The paper addresses early diagnostic stratification for Wireless Body Area Networks (WBANs) where lightweight, robust edge inference is critical.

3. **Reproducibility & Practical Feasibility:**
   The paper provides clear hyperparameter configurations (Table 3), transparent data preprocessing steps (Min-Max scaling, median/mode imputation, stratified 80/20 train-test splitting), and benchmark metrics (Table 4 & 5). It relies entirely on a publicly accessible benchmark dataset, making complete reproduction and empirical critique feasible without proprietary computational barriers.

---
**Student Signature:** ___________________________  
**Approval Status:** [  ] Approved   [  ] Revisions Requested   [  ] Rejected  
**Instructor Feedback:** _________________________________________________________________
