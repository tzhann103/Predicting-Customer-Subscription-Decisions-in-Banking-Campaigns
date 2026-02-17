# Predicting Term Deposit Subscription (Bank Marketing)

## Overview
Retail banks use outbound campaigns to sell term deposits, but call capacity is limited and broad targeting wastes effort. This project builds predictive models to estimate subscription likelihood, rank customers, and support more efficient outreach.

## Data
- UCI Bank Marketing Dataset(Portugal)
- 45,211 records, 17 variables, target `y` (subscribe: yes/no)
- Highly imbalanced outcome (~11.7% “yes”)
- Modeling note: `duration` excluded for realistic pre-call prediction

## Methods
Preprocessing:
- Kept “unknown” as a valid category (no row deletion)
- One-hot encoding for categorical variables; binary mapping for yes/no fields
- 80/20 stratified train-test split; cross-validation for robustness

Models:
- Logistic Regression, Lasso LR, Random Forest, XGBoost, LightGBM, Neural Network

## Results (Test Set)
| Model | ROC-AUC | Precision (pos) | Recall (pos) | F1 (pos) |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.772 | 0.27 | 0.62 | 0.37 |
| Lasso LR | 0.773 | 0.27 | 0.64 | 0.38 |
| Random Forest | 0.802 | 0.43 | 0.55 | 0.48 |
| XGBoost | 0.805 | 0.35 | 0.65 | 0.45 |
| LightGBM | **0.806** | 0.35 | **0.65** | 0.45 |
| Neural Network | 0.788 | 0.61 | 0.24 | 0.24 |

## Business Implementation
- Score and rank customers by predicted probability
- Contact top-ranked customers within capacity constraints
- Use SHAP to guide practical rules (e.g., prioritize prior campaign success/high balance; manage repeated contact attempts; treat `contact_unknown` as a data-quality flag)

## Tools
Python: scikit-learn, TensorFlow/Kers, SHAP
