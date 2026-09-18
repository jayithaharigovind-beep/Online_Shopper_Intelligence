# Online Shopper Intelligence

## Machine Learning-Based Purchase Intention Prediction

### Project Overview

Online Shopper Intelligence is a machine learning project developed to predict whether an online visitor is likely to make a purchase during a browsing session.

The project uses historical online shopping session data to identify behavioural and contextual patterns associated with purchase intention. Multiple classification models were evaluated, and Random Forest was selected for deployment based on its balance between identifying potential purchasers and overall classification performance.

The trained model has been integrated into an interactive Streamlit dashboard that provides purchase probability and purchase-potential classification.

## Business Problem

E-commerce websites receive a large number of visitors, but only a proportion of them complete a purchase.

The objective is to identify visitors who show stronger purchase-intention signals so that businesses can prioritize engagement, personalization, and conversion efforts.

## Dataset

The project uses the Online Shoppers Intention dataset.

- Total sessions: 12,330
- Purchase sessions: 1,908
- Purchase rate: approximately 15.5%
- Target variable: Revenue

Revenue is a binary target:

- 1 = Purchase
- 0 = No Purchase

Each row represents an individual online shopping session.

## Features Used

The final model uses 17 input features.

### Numerical Features

- Administrative
- Administrative_Duration
- Informational
- Informational_Duration
- ProductRelated
- ProductRelated_Duration
- TotalPages
- BounceRates
- ExitRates
- SpecialDay
- OperatingSystems
- Browser
- Region

### Categorical Features

- TrafficType
- VisitorType
- Weekend
- Month

### Feature Engineering

A new feature called TotalPages was created:

TotalPages = Administrative + Informational + ProductRelated

Revenue was used as the target variable and was not included as an input feature.

PageValues was excluded from the final model because it can contain information closely associated with the purchase outcome and may create leakage or unrealistic prediction conditions for prospective prediction.

## Machine Learning Workflow

Data Collection → Exploratory Data Analysis → Class Imbalance Analysis → Train-Test Split → Data Preprocessing → Feature Engineering → Model Training → Model Comparison → Model Evaluation → Feature Importance Analysis → Model Saving → Streamlit Deployment

## Class Imbalance

The dataset contains significantly more non-purchase sessions than purchase sessions.

- 84.5% = No Purchase
- 15.5% = Purchase

Because of this imbalance, accuracy alone is not sufficient for evaluating the model.

The Random Forest model uses class_weight="balanced" to account for the unequal class distribution during training.

## Models Evaluated

Five classification models were evaluated:

1. Logistic Regression
2. Decision Tree
3. AdaBoost
4. Gradient Boosting
5. Random Forest

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 62.98% | 25.93% | 74.87% | 38.52% | 0.7347 |
| Decision Tree | 63.95% | 25.93% | 71.47% | 38.05% | 0.7319 |
| AdaBoost | 84.47% | 0.00% | 0.00% | 0.00% | 0.7345 |
| Gradient Boosting | 84.27% | 44.83% | 6.81% | 11.82% | 0.7595 |
| Random Forest | 70.64% | 29.50% | 64.40% | 40.46% | 0.7522 |

## Model Selection

Random Forest was selected for deployment based on the project's emphasis on identifying potential purchasers.

The Random Forest model achieved:

- 64.40% recall
- 29.50% precision
- 0.4046 F1 score
- 0.7522 ROC-AUC

Although AdaBoost and Gradient Boosting produced higher accuracy, their recall for purchasers was extremely low in this evaluation. Therefore, accuracy alone was not used as the model selection criterion.

## Model Explainability

Random Forest feature importance was used to understand which variables were most influential in the model's predictions.

| Feature | Importance |
| --- | ---: |
| ExitRates | 17.26% |
| ProductRelated_Duration | 12.63% |
| BounceRates | 8.33% |
| TotalPages | 7.77% |
| ProductRelated | 7.66% |

TotalPages ranked fourth in feature importance at approximately 7.77%.

Feature importance represents predictive contribution within the model and does not establish causation.

## Streamlit Dashboard

The trained model has been deployed through an interactive Streamlit dashboard.

The dashboard includes:

- Executive Overview
- Purchase Prediction
- Dataset Diagnostics
- Model Performance
- Model Explainability
- Business Insights

The Purchase Prediction section accepts visitor session characteristics and provides a purchase probability, purchase-potential category, and business interpretation.

## MLOps Considerations

If deployed as a production ML system, the model would require continuous monitoring and governance.

Key considerations include:

- Data drift
- Concept drift
- Model performance monitoring
- Data quality validation
- Model versioning
- Periodic retraining
- Access control and data privacy
- Auditability and governance
- Human oversight

The model should be used as a decision-support system rather than as a completely autonomous system for customer-related decisions.

## Deployment Architecture

Online Shopper Session Data → Data Preprocessing → Feature Engineering → Random Forest Model → Purchase Probability → Streamlit Dashboard → Business Decision Support

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Matplotlib
- Seaborn
- Streamlit
- Jupyter Notebook
- GitHub

## Project Files

- app.py
- online_shoppers_intention.csv
- online_shoppers_rf_pipeline.pkl
- online_shoppers_ml.ipynb
- requirements.txt
- README.md

## Live Dashboard

https://onlineshopperintelligence-nz725n23yffmo5xxltpnmj.streamlit.app

## GitHub Repository

https://github.com/jayithaharigovind-beep/Online_Shopper_Intelligence

## Conclusion

The Online Shopper Intelligence project demonstrates how machine learning can be applied to online shopping session data to estimate purchase intention.

The project compares multiple classification algorithms, addresses class imbalance, performs feature engineering, evaluates model performance using multiple metrics, and uses Random Forest feature importance for model interpretation.

The final model is integrated into a Streamlit dashboard that converts model predictions into an interactive business decision-support tool.

The project also considers production-level MLOps requirements including monitoring, data drift, concept drift, model versioning, retraining, governance, and responsible deployment.
