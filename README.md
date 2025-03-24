AI / Machine Learning Components Used
This project leverages several machine learning models and AI techniques to predict career aspirations based on user input data. Below is a breakdown of the AI/ML components:

Models
RandomForestClassifier (from scikit-learn)

Type: Ensemble Learning (Bagging)

Purpose: Predicts career_aspiration by building multiple decision trees and combining their results for improved accuracy and generalization.

XGBClassifier (from XGBoost)

Type: Gradient Boosted Decision Trees

Purpose: Advanced model optimized for performance on structured data, widely used in real-world AI tasks and competitions.

Data Processing & AI Techniques
SMOTE (Synthetic Minority Over-sampling Technique)

Balances the dataset by generating synthetic samples of the minority class to improve model fairness and accuracy.

StandardScaler

Scales numerical features to have zero mean and unit variance, helping models train more efficiently.

LabelEncoder

Encodes categorical target labels (career_aspiration) into numeric form for model training.

Model Evaluation
Classification Report (Precision, Recall, F1-Score, Support)

Evaluates the performance of both models to ensure accuracy and balanced predictions.

Model Persistence
joblib

Saves trained models, scalers, and label encoders as .pkl files for future predictions and deployment.

Summary
This AI module builds and evaluates predictive models capable of recommending career paths based on user-provided data. Techniques like Random Forest, XGBoost, SMOTE balancing, and feature scaling ensure that the system is robust, fair, and ready for real-world deployment.

