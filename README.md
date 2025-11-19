# 🌟 Metastatic Cancer Diagnosis Prediction

Predicting durations for metastatic cancer diagnosis to uncover patterns in healthcare equity.


## 🧭 Overview

This project was developed for the 2024 WiDS Datathon (Challenge 2) sponsored by Gilead Sciences, using a large-scale dataset enriched with demographic, socioeconomic, and climate data.

The goal is to predict how long it takes for a patient to receive a metastatic cancer diagnosis, using this duration as a proxy for healthcare access and equity.

Metastatic TNBC is highly aggressive, and early diagnosis is essential. A model that highlights disparities in diagnosis wait time can support targeted interventions and more equitable patient outcomes.

## 🚀 Key Highlights

- Random Forest model achieved an R² score of 0.73 with an MSE of 0.12.

- Lasso Regression was used to identify the top 20 most predictive features.

- Regional, socioeconomic, and climate variables show measurable influence on diagnosis delays.

- Full ML pipeline: EDA → preprocessing → feature engineering → selection → modeling → evaluation.

## 🗂️ Repository Structure
```
Metastatic-Cancer-Prediction/
├── notebooks/              
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── src/                    
│   ├── preprocess.py
│   ├── feature_select.py
│   ├── train_model.py
│   └── evaluate.py
├── results/                
│   ├── feature_importance.png
│   ├── model_scores.png
│   └── predictions_hist.png
├── requirements.txt
└── README.md
```

## ⚙️ Setup & Installation
1. Clone the Repository
```
git clone https://github.com/Anni-Bamwenda/Metastatic-Cancer-Prediction.git
cd Metastatic-Cancer-Prediction
```

2. Install Dependencies
```
pip install -r requirements.txt
```

3. Run the Notebook

```
jupyter notebook notebooks/anni-widsdatathon02.ipynb
```

## 🔍 Exploratory Data Analysis (EDA)

The dataset provided by HealthVerity + climate enrichment includes:

- Demographics

- Cancer diagnosis/treatment codes

- Insurance & socioeconomic indicators

- ZIP-level temperature patterns

## 📊 Visualizations

### Diagnosis Counts by Race
![Images/No. of patients by race img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/No.%20of%20patients%20by%20race%20img.png)

### Age Distribution

![Images/Patient Age Distribution img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Patient%20Age%20distribution%20img.png)

<!---
![No. of patients with 0days diagnosis period img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/No.%20of%20patients%20with%200days%20diagnosis%20period%20img.png)


![Images/Types of payment by patient race img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Types%20of%20payment%20by%20patient%20race%20img.png)
--->

## 🛠️ Data Preprocessing

Key preprocessing steps:

- Dropped irrelevant columns (e.g., gender — all patients are women).

- Removed duplicates and handled missing values.

- Replaced outliers in geographic, climate, and diagnosis code fields.

- Standardized all numerical features.

- Label-encoded categorical variables.

- Created age group bins for improved interpretability.

## 🧬 Feature Selection (Lasso Regression)

Lasso (L1 regularization) was used to identify the top 20 predictive features, with hyperparameters tuned via GridSearchCV.

### Top 20 Feature Coefficients

![Images/Top 20 Features by coeff. value img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Top%2020%20Features%20by%20coeff.%20value%20img.png)

Feature selection improved:

- Model Interpretability

- Training Performance

- Resistance to overfitting
  
- Dimensionality Reduction

## 🌲 Modeling

Two models were compared:

| Model            | R² Score |   MSE   | Notes                |
|------------------|---------:|--------:|----------------------|
| Random Forest    | **0.73** | **0.12** | Selected final model |
| Lasso Regression | 0.58     | 0.19     | Baseline model       |

The Random Forest captured nonlinear relationships and outperformed the linear model.

## 📈 Predictions

A distribution of predicted diagnosis durations shows clustering in the 0 –100 day range, highlighting regions with potentially acceptable care and others that need improvement.

### Sample predictions (CSV available in results/).

![Images/Sample Predictions img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Sample%20Predictions%20img.png)

### Predictions Histogram

![Images/Predictions histogram img.png](https://github.com/Anni-Bamwenda/WidsDatathon/blob/main/Images/Predictions%20histogram%20img.png)

## 📝 Notes

Potential improvements for future iterations:

- Hyperparameter tuning (RandomizedSearch / Bayesian Optimization)

- Add ensemble models (XGBoost, LightGBM)

- Add SHAP values for interpretability

- Deploy model via FastAPI + Docker

- Add pytest unit tests for feature preparation and model pipeline

## 🧰 Tech Stack

- Python 3.10+

- NumPy, Pandas

- Matplotlib, Seaborn

- scikit-learn

- Developed in Kaggle Notebooks

## 👩🏽‍💻 Author

Anni Bamwenda
Software Engineer II • Data Scientist • AI/ML Engineer

🔗 LinkedIn https://www.linkedin.com/in/annibamwenda/

🔗 GitHub: https://github.com/Anni-Bamwenda
