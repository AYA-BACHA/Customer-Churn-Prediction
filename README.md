# Customer Churn Prediction — Djezzy Internship

> An end-to-end machine learning project developed during my internship at **Djezzy Algeria** to predict customer churn and transform model predictions into actionable customer-risk states.

## Overview

Customer churn is an important challenge for telecom companies. If customers at risk of leaving can be identified early, the company can potentially take targeted retention actions.

During my internship at Djezzy, I worked on a **customer churn prediction system** designed to answer:

> **"Which customers are likely to leave, and which customers should the company pay attention to?"**

The project goes beyond simply predicting `Churn = Yes/No`.

The final solution transforms the prediction into five business-oriented customer states:

**Left → May Leave → Stable → May Become Loyal → Loyal**

This makes the machine learning output easier to understand and potentially more useful for a business user.

---

## Project Workflow

The project follows an end-to-end machine learning workflow:

 
Customer Data
     ↓
Data Cleaning & Preprocessing
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Model Training & Comparison
     ↓
Model Evaluation
     ↓
Probability Threshold Tuning
     ↓
Customer Risk Classification
     ↓
Business-Oriented Customer States
     ↓
Prediction Platform
  

---

## Business Objective

The objective was to build a system capable of identifying customers who may be at risk of churn.

In a churn-retention scenario, **false negatives** are particularly important.

A false negative occurs when:

> The model predicts that a customer will stay, but the customer actually churns.

Missing such a customer means the company may lose the opportunity to intervene.

For this reason, the project places strong emphasis on **churn recall**, while still monitoring accuracy, precision, and F1-score.

---

# Dataset

The dataset contains **7,043 customer records** and **21 variables** before preprocessing.

It includes information such as:

* Customer demographics
* Tenure
* Contract type
* Phone and internet services
* Additional subscribed services
* Payment method
* Monthly charges
* Total charges
* Churn status

The target variable is:

 
No  → 0
Yes → 1
  

### Data preprocessing

`TotalCharges` was converted to a numerical variable, with invalid values identified as missing values.

After preprocessing, the final dataset contained:

**7,032 customers**

with the cleaned data used for analysis and modeling.

> The original customer dataset is not included in this repository.

---

# Exploratory Data Analysis

The analysis focused on understanding which customer characteristics were associated with churn.

Some of the main variables investigated were:

### Tenure

Customer tenure was analyzed to understand how churn behavior changes depending on how long a customer has been with the company.

### Monthly Charges

Monthly charges were examined as a potential indicator of customer churn behavior.

### Contract Type

Contract type was one of the key variables explored because customers under different contract structures can exhibit different churn patterns.

Other service-related variables were also considered during the analysis.

---

# Feature Engineering

Additional features were created to provide the models with more useful information.

Examples include:

### `tenure_band`

Groups customers into different tenure ranges.

### `charges_per_month`

A derived feature based on customer charge information.

### `num_services`

Represents the number of subscribed services associated with a customer.

Categorical variables were also encoded so that they could be used by machine learning models.

---

# Model Development

Several classification approaches were explored during the project, including:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Ensemble approaches
* Stacking

The models were evaluated using several metrics rather than relying only on accuracy.

The main metrics considered were:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

# Model Evaluation

A Logistic Regression baseline achieved:

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **80.38%** |
| Churn Precision |    **65%** |
| Churn Recall    |    **57%** |
| Churn F1-score  |    **61%** |

The Random Forest and ensemble experiments were then used to explore whether the model could identify a larger proportion of customers who actually churn.

The final operating configuration focused on improving **churn recall**.

---

# Probability Threshold Tuning

A classification model does not have to use `0.50` as its probability threshold.

Instead, different thresholds were tested to understand the trade-off between:

* identifying more customers who churn,
* generating more false positives,
* and maintaining reasonable overall performance.

### Threshold comparison

| Threshold |   Accuracy | Churn Recall |  Precision |   F1-score |
| --------: | ---------: | -----------: | ---------: | ---------: |
|      0.30 |     67.10% |       91.20% |     44.20% |     59.60% |
|      0.35 |     69.50% |       86.40% |     46.00% |     60.00% |
|  **0.40** | **71.82%** |   **85.29%** | **48.26%** | **61.64%** |
|      0.45 |     73.70% |       82.90% |     50.30% |     62.60% |

### Final operating threshold: `0.40`

A threshold of **0.40** was used for the final operating configuration.

At this threshold:

* **Accuracy:** 71.82%
* **Churn Recall:** 85.29%
* **Churn Precision:** 48.26%
* **F1-score:** 61.64%

The choice reflects the project's business objective of detecting a high proportion of customers at risk of churn, while accepting that this produces more false positives.

---

# Confusion Matrix

At the final threshold of `0.40`:

 
                    Predicted
                  Stay    Churn
Actual Stay        693     342
Actual Churn        55     319
  

This means the model correctly identified **319 customers who churned**, while **55 actual churners were missed**.

At the same time, **342 customers who stayed were classified as potential churners**, illustrating the precision/recall trade-off.

---

# From Churn Probability to Business States

One of the main goals of the project was to make the ML output easier to understand for a non-technical user.

Instead of displaying only a probability such as:

 
Churn probability = 0.73
  

the platform translates customer predictions into five business-oriented states:


Left
May Leave
Stable
May Become Loyal
Loyal


These states are generated as part of the prediction workflow using the customer's churn-related information and the model's prediction, with each customer assigned to **one business state**.

The five states provide a simpler way of communicating the customer's situation to someone using the platform.

---

# Customer State Distribution

After applying the customer-state classification to the **7,032 customers** in the processed dataset, the distribution was:

| Customer State       | Number of Customers | Percentage |
| -------------------- | ------------------: | ---------: |
| **Left**             |               1,044 |     14.85% |
| **May Leave**        |               1,824 |     25.94% |
| **Stable**           |               1,518 |     21.59% |
| **May Become Loyal** |                 843 |     11.99% |
| **Loyal**            |               1,803 |     25.64% |
| **Total**            |           **7,032** |   **100%** |

This gives the platform a more intuitive customer-level view instead of presenting only raw model predictions.

### Why these states?

The purpose of the five-state system is to create a gradual interpretation of customer status:

* **Left** → customers identified as having already left
* **May Leave** → customers showing higher churn risk
* **Stable** → customers with a relatively stable situation
* **May Become Loyal** → customers showing lower churn risk and positive signals
* **Loyal** → customers showing the strongest retention-related signals

The model's churn prediction is therefore converted into a **business-facing interpretation** rather than being presented as a raw machine learning output.

---

# Prediction Platform

A prediction platform was developed to demonstrate how the model could be used in practice.

The user can enter a **Customer ID** and obtain information about that customer, including:

* Customer state
* Churn prediction
* Risk-related information
* The factors used to explain the prediction

The objective was to bridge the gap between a machine learning model and a simple interface that a business user could understand.

---

# Example

Instead of requiring a user to interpret a machine learning probability manually:

 
Customer ID
     ↓
Model Prediction
     ↓
Churn Probability
     ↓
Business Classification
     ↓
"May Leave"
  

The platform provides a more understandable representation of the prediction.

---

# Technologies

### Data & Machine Learning

* Python
* Pandas
* NumPy
* Scikit-learn
* XGBoost

### Data Visualization

* Matplotlib
* Seaborn

### Development

* Jupyter Notebook
* Python

---

# What I Learned

This project gave me practical experience with the complete machine learning pipeline, including:

* Data cleaning
* Handling missing values
* Exploratory data analysis
* Feature engineering
* Categorical encoding
* Classification algorithms
* Ensemble learning
* Model comparison
* Stratified evaluation
* Confusion matrices
* Precision vs. recall
* Probability threshold tuning
* Translating ML predictions into business concepts
* Building a prediction interface

One of the most important lessons was that **the model with the highest accuracy is not necessarily the best configuration for a specific business objective**.

The appropriate evaluation depends on what the company is trying to achieve.

---

# Project Highlights

### End-to-end ML workflow

From raw customer data to a usable prediction system.

### Business-oriented modeling

The project considers the consequences of false negatives and false positives instead of optimizing accuracy alone.

### Threshold optimization

The classification threshold was explicitly tested and adjusted according to the project's churn-detection objective.

### Five-level customer classification

Model predictions were translated into:

**Left · May Leave · Stable · May Become Loyal · Loyal**

### Application layer

The final model was integrated into a customer prediction platform to demonstrate how the ML workflow could be used by a non-technical user.

---

# Internship Context

This project was developed during my internship at **Djezzy Algeria**.

The internship provided an opportunity to apply machine learning concepts to a real business-oriented problem and to work through the process of turning customer data into a predictive solution.

The project combines:

**Data Analysis + Machine Learning + Business Reasoning + Application Development**

---

# Disclaimer

This repository is intended for **educational and portfolio purposes**.

The dataset used for experimentation is a publicly available telecom churn dataset and does not contain confidential Djezzy customer information.

The model is a predictive prototype and should not be considered a production-ready customer-retention system.

---

## Author

**AYA**
Computer Science Engineering Student — ESI Algiers

Interested in **Machine Learning, Data Science, and AI**.
