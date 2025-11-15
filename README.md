# Diabetes-Classification.
🩺 Diabetes Classification Using the Sowutuom Clinic Dataset
📘 Overview

This project uses a synthetic medical dataset from Sowutuom Clinic to build machine learning models that classify patients as diabetic (1) or non-diabetic (0).

Two ML models were developed and compared:

Logistic Regression

Random Forest Classifier

The goal is to build a reliable classifier, evaluate its performance, and identify the most influential features.

📂 Dataset Description

Name: Sowutuom Clinic Dataset
Rows: 500
Columns: 9
Source: Synthetic
Purpose: Diabetes prediction, BMI/glucose analysis, genotype patterns

🧬 Fields
Column	Type	Description
clinic	Categorical	clinic_1 – clinic_10
age	Integer	18–90
height	Float (cm)	Patient height
weight	Float (kg)	Patient weight
bmi	Float	Weight / height²
glucose_level	Integer	Fasting glucose (mg/dL)
blood_group	Categorical	A+, A−, B+, B−, AB+, AB−, O+, O−
genotype	Categorical	AA, AS, SS
diabetic	Integer	0 = non-diabetic, 1 = diabetic
✔️ Integrity Checks

No missing values

Correct data types

BMI values match formula

Glucose values in realistic range

🎯 Objectives

Preprocess medical data

Build two ML models

Evaluate Accuracy, Precision, Recall, F1-Score

Identify most influential predictors

Compare model performance

Produce a reproducible Google Colab workflow

🛠️ Tech Stack
Category	Tools
Language	Python
Data Handling	Pandas, NumPy
Visualization	Matplotlib, Seaborn
Machine Learning	scikit-learn
Environment	Google Colab
🔍 Project Workflow
1️⃣ Load and explore data

Load CSV with Pandas

Show dataset info, summary stats, and missing values

2️⃣ Preprocess

Encode categorical variables (clinic, blood_group, genotype)

Prepare feature matrix (X) and label vector (y)

3️⃣ Train/Test Split

80% training, 20% testing

Ensures fair evaluation and prevents overfitting

4️⃣ Train Models

Logistic Regression

Random Forest Classifier (200 trees)

5️⃣ Evaluate Performance

Metrics used:

Accuracy

Precision

Recall

F1-Score

Classification Report

6️⃣ Feature Importance

Random Forest identifies the most important predictors:

Glucose Level

BMI

Age

These strongly influence diabetes prediction.

7️⃣ Model Comparison

Random Forest consistently outperformed Logistic Regression.

🏆 Best Model
⭐ Random Forest Classifier

Why?

Handles non-linear patterns

Higher accuracy

Higher recall (= catches more diabetic cases)

Better F1-score

Shows feature importance

📈 Key Results
Model	Accuracy	Precision	Recall	F1-Score
Logistic Regression	Good	Moderate	Lower	Moderate
Random Forest	Great	High	High	High

Random Forest is the recommended model for deployment.

📁 Recommended Folder Structure
📦 diabetes-classification
│
├── README.md
├── sowutuom_clinic_dataset.csv
├── diabetes_classification.ipynb
│
├── images/
│   └── feature_importance.png
│
└── results/
    ├── logistic_regression_report.txt
    ├── random_forest_report.txt
    └── metrics_summary.csv

🚀 How to Run the Project (Google Colab)

Upload sowutuom_clinic_dataset.csv to Colab

Upload the provided notebook

Run cells step-by-step

View evaluation scores and graphs

Modify hyperparameters to improve accuracy

📌 Possible Extensions

Add more models (XGBoost, SVM, Neural Network)

Hyperparameter tuning using GridSearchCV

Deploy using Streamlit or Flask

Add confusion matrix & ROC-AUC visualization

Build a real-time API

👨‍💻 Author

Steven Tesla
IT Student • Cybersecurity & ML Enthusiast
Pentecost University

📧 Email: steventesla756@gmail.com

🐙 GitHub: https://github.com/Steven25
