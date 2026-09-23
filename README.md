# Customer Churn Prediction — Djezzy

> Machine learning project developed during my internship at **Djezzy Algeria** to identify customers who are at risk of churning and support customer-retention decisions.

## Overview

Customer churn is a major challenge for telecom companies: identifying customers who may leave early can give the company an opportunity to take action and improve retention.

This project develops a **customer churn prediction system** that estimates the probability that a customer will leave the service.

The project covers the complete ML workflow:

**Data → Analysis → Feature Engineering → Model Training → Evaluation → Threshold Selection → Customer Risk Classification → Prediction Platform**

## Business Objective

The main goal is not simply to maximize accuracy.

For a churn-prevention system, **missing a customer who is actually going to leave (False Negative)** can be costly because the company loses the opportunity to intervene.

Therefore, the project places particular attention on **churn recall**, while still considering precision and overall accuracy.

## Dataset

The project uses a telecom customer dataset containing information such as:

* Customer demographics
* Tenure
* Contract type
* Internet and phone services
* Additional subscribed services
* Payment method
* Monthly charges
* Total charges
* Churn status

The target variable is:

* `0` → Customer stays
* `1` → Customer churns

> **Note:** The original customer dataset is not included in this public repository.

## Exploratory Data Analysis

The analysis focused on identifying patterns associated with customer churn.

Some of the most important factors explored were:

* **Tenure**
* **Monthly Charges**
* **Contract Type**
* Number of subscribed services
* Customer service characteristics

The analysis showed that churn behavior is not uniform across customers and that several customer characteristics can provide useful predictive signals.

## Feature Engineering

Additional features were explored to provide the models with more useful information, including:

* `tenure_band` — groups customers according to their tenure
* `charges_per_month` — derived from customer charge information
* `num_services` — number of subscribed services

Categorical variables were also encoded so they could be used by the machine learning models.

## Models

Several classification approaches were explored and compared during the project, including:

* Logistic Regression
* Decision Tree
* Random Forest
* Gradient Boosting
* XGBoost
* Ensemble approaches

The experiments helped evaluate the trade-off between **accuracy, precision, recall, and F1-score**.

## Final Model

The final solution uses a **Random Forest classifier**.

Instead of automatically using the default probability threshold of `0.50`, the prediction threshold was adjusted according to the business objective.

### Operating threshold: `0.40`

At this threshold, the model achieved:

| Metric          |     Result |
| --------------- | ---------: |
| Accuracy        | **71.82%** |
| Churn Recall    | **85.29%** |
| Churn Precision | **48.26%** |
| Churn F1-score  | **61.64%** |

### Confusion Matrix

|                  | Predicted Stay | Predicted Churn |
| ---------------- | -------------: | --------------: |
| **Actual Stay**  |            693 |             342 |
| **Actual Churn** |             55 |             319 |

The threshold was selected to prioritize identifying a larger proportion of customers who actually churn, accepting a higher number of false positives in the process.

## Why Recall Matters

In this business context, a **False Negative** means:

> The model predicts that a customer will stay, but the customer actually churns.

These customers can be missed by a retention campaign.

A **False Positive** means:

> The model predicts that a customer may churn, but the customer actually stays.

This can lead to unnecessary retention actions.

The threshold therefore represents a **business trade-off**, rather than simply a mathematical optimization.

## Customer Risk Classification

The prediction system can translate the model's output into customer states such as:

* **Left**
* **May Leave**
* **Stable**
* **May Become Loyal**
* **Loyal**

The goal is to make the prediction easier to interpret for a non-technical user rather than displaying only a probability.

For each prediction, the platform can also provide the main factors considered when determining the customer's state.

## Prediction Platform

A simple interface was developed to demonstrate how the model could be used in practice.

The user can provide a **customer ID** and receive:

* Predicted customer state
* Churn risk
* Supporting factors
* Prediction information

The interface was designed around a **Djezzy-inspired visual identity** to demonstrate how the ML model could be integrated into a business-facing application.

*The exact structure may vary depending on the final organization of the repository.*

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook
* Machine Learning
* Data Analysis
* Classification

## Key Takeaways

This project helped me work through a complete applied machine learning workflow, from raw customer data to a business-oriented prediction system.

In particular, I gained practical experience with:

* Data cleaning and preprocessing
* Exploratory data analysis
* Feature engineering
* Classification models
* Ensemble learning
* Model evaluation
* Confusion matrices
* Precision vs. recall trade-offs
* Probability threshold tuning
* Translating ML predictions into business-oriented decisions
* Integrating a machine learning model into a simple application

## Internship Context

This project was developed during my internship at **Djezzy Algeria**, where I worked on applying machine learning to a customer churn prediction problem.

The project combines **data analysis, machine learning, and application development** to demonstrate how predictive models can support customer-retention strategies.

## Disclaimer

This repository is intended for **educational and portfolio purposes**.

The dataset used for experimentation is a publicly available telecom churn dataset and does not contain confidential Djezzy customer information.

The project is a predictive prototype and should not be interpreted as a production-ready churn management system.

---

**Author:** AYA
**Computer Science Engineering Student — ESI Algiers**
