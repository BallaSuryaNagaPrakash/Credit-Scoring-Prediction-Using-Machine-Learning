# Credit-Scoring-Prediction-Using-Machine-Learning
Machine learning model to predict whether a loan applicant is creditworthy or at risk of default. Includes data analysis, feature engineering, model training, and performance evaluation.

# Credit Scoring Model — Machine Learning Project

## Project Overview
This project builds a machine learning model to predict the creditworthiness of loan applicants using financial and behavioral data. Credit scoring is widely used by banks, fintech companies, and lending institutions to evaluate whether a borrower is likely to repay a loan or default.

The project demonstrates a complete end-to-end machine learning workflow including data generation, exploratory data analysis (EDA), feature engineering, model training, and model evaluation.

--------------------------------------

## Business Problem
Financial institutions must evaluate loan applications carefully to reduce the risk of loan defaults. Manual evaluation is time-consuming and may introduce bias.

Machine learning models can analyze financial indicators such as income, credit score, and debt levels to automatically classify applicants as **creditworthy** or **high risk**.

---

## Problem Statement
Develop a machine learning model that predicts whether a loan applicant is creditworthy based on their financial history and behavioral attributes.

The model classifies applicants into two categories:

- Creditworthy (Low Risk)
- Default Risk (High Risk)

---

## Project Objectives
- Perform Exploratory Data Analysis (EDA) to understand financial data patterns
- Engineer meaningful financial features
- Train multiple machine learning classification models
- Compare model performance using evaluation metrics
- Identify key factors influencing credit risk

---

## Dataset
The dataset contains **1000 synthetic loan applicant profiles** with financial and behavioral attributes.

### Key Features

| Feature | Description |
|------|------|
| Age | Age of the applicant |
| Annual Income | Yearly earnings |
| Employment Years | Years of employment |
| Loan Amount | Requested loan amount |
| Loan Term | Loan repayment period |
| Credit Score | Creditworthiness score |
| Credit Utilization | Percentage of credit limit used |
| Existing Debt | Current outstanding loans |
| Late Payments | Number of missed payments |
| Debt to Income Ratio | Debt relative to income |

---

## Exploratory Data Analysis
EDA was performed to understand feature distributions and relationships between variables.

Key insights:
- Credit score strongly influences loan repayment probability
- Higher credit utilization increases default risk
- Borrowers with many late payments have higher risk
- Debt-to-income ratio impacts repayment ability

Visualizations used:
- Distribution plots
- Box plots
- Correlation heatmap
- Feature relationship analysis

---

## Machine Learning Models
The following classification models were trained and evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

---

## Model Evaluation Metrics
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix

---

## Results
Ensemble models such as **Random Forest and Gradient Boosting** showed better performance compared to single models.

Important predictive features include:

- Credit Score
- Number of Late Payments
- Credit Utilization
- Debt-to-Income Ratio

---

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Project Structure
