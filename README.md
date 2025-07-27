# Blood Pressure Anomaly Prediction

This repository contains a machine learning workflow to predict abnormal blood pressure based on patient data. It includes data preprocessing, feature engineering, and classification modeling using Decision Tree and Logistic Regression, with class imbalance handled by SMOTE oversampling.



## Project Overview

This project analyzes patient blood pressure readings to classify blood pressure as normal or abnormal. Missing values are imputed using patient-specific or overall means, and outliers are filtered to focus on meaningful data ranges.

## Data Description

- `fake_LCICM_bp_data.csv`: Synthetic per-patient blood pressure measurements.
- `fake_LCICM_labels.csv`: Corresponding binary labels indicating abnormal (1) or normal (0) blood pressure.

## Data Preprocessing

- Missing blood pressure values filled using the patient's mean or overall mean.
- Data aggregated to average blood pressure per patient.
- Outliers beyond ±2 standard deviations removed.
- Dataset merged with labels for supervised learning.

## Feature Engineering

- Feature: Average blood pressure per patient.
- Target: Binary label (0: normal, 1: abnormal).

## Modeling

### Decision Tree with SMOTE

- Combines SMOTE oversampling and a decision tree classifier in a pipeline.
- Uses cross-validated grid search tuning with an entropy split criterion.
- Evaluated via stratified 5-fold cross-validation and a held-out test set.

### Logistic Regression with SMOTE

- Applies SMOTE for balancing, followed by feature scaling.
- Trains logistic regression as a baseline classifier.
- Evaluated on the same train/test split for consistent comparison.

## Hyperparameter Tuning

- GridSearchCV optimizes decision tree parameters (max depth, leaf size, etc.) to minimize entropy.
- Best model achieved ~71% cross-validation accuracy.

## Evaluation

The classification reports include precision, recall, F1-score, and support for each class. Results show:

- Decision tree generally outperforms logistic regression on this task.
- SMOTE oversampling improves minority class prediction.
- Example test set metrics for decision tree:
  - Precision (class 0): 0.76
  - Recall (class 1): 0.67
  - Accuracy: ~55%
- Example metrics for logistic regression:
  - Precision (class 0): 0.71
  - Recall (class 1): 0.40
  - Accuracy: ~60%


