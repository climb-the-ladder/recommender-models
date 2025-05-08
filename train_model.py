import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.normpath(os.path.join(script_dir, "../recommender-data/processed/processed_dataset.csv"))

# Load the preprocessed dataset
df = pd.read_csv(csv_path)

# Features and Target
X = df.drop(columns=["career_aspiration"])
y = df["career_aspiration"]

# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Apply SMOTE for balancing
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# ================= WHY XGBOOST? =================
# XGBoost is chosen as the primary model for several reasons:
# 1. Performance: XGBoost is consistently one of the top-performing algorithms for tabular data, especially in classification tasks like this one.
# 2. Handling of Imbalanced Data: With built-in support for class weighting and robust objective functions, XGBoost can handle imbalanced datasets well (even though we also use SMOTE).
# 3. Speed and Scalability: XGBoost is highly optimized for speed and memory efficiency, making it suitable for both small and large datasets.
# 4. Regularization: It includes L1 and L2 regularization, which helps prevent overfitting and improves generalization.
# 5. Feature Importance: XGBoost provides clear feature importance metrics, aiding in model interpretability and feature selection.
# 6. Flexibility: It supports various objective functions and evaluation metrics, and can be tuned extensively for optimal performance.
# 7. Community and Support: XGBoost is widely used in industry and academia, with extensive documentation and community support.
# 
# Compared to alternatives:
# - Random Forests are strong but often less accurate and slower for large, complex datasets.
# - LightGBM and CatBoost are competitive, but XGBoost is more mature and widely adopted.
# - SVMs and Logistic Regression are less effective for complex, high-dimensional, or non-linear data.
# - Neural Networks (MLP) require more data and tuning, and are less interpretable for tabular data.
# 
# For these reasons, XGBoost is the best fit for this recommender/classification task.
# ================= END WHY XGBOOST =================
# XGBoost Classifier
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    objective='multi:softmax',
    num_class=len(label_encoder.classes_),
    eval_metric='mlogloss',
    random_state=42
)
xgb_model.fit(X_train_scaled, y_train)

# ================= ALTERNATIVE MODELS (Commented Out) =================
# The following are alternative models we could use for this classification task.
# They are commented out, but included for reference and discussion.

# --- LightGBM Classifier ---
# from lightgbm import LGBMClassifier
# # LightGBM is fast and efficient for large datasets, and often performs well on tabular data.
# # It supports categorical features natively (with 'categorical_feature' param), but here we use scaled data.
# # Common hyperparameters to tune: n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, subsample, colsample_bytree.
# lgbm_model = LGBMClassifier(
#     n_estimators=300,
#     learning_rate=0.05,
#     max_depth=8,
#     num_leaves=31,  # Increase for more complex data
#     min_child_samples=20,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     class_weight='balanced',
#     random_state=42,
#     n_jobs=-1
# )
# lgbm_model.fit(X_train_scaled, y_train)
# # Evaluation:
# # print(classification_report(y_test, lgbm_model.predict(X_test_scaled), target_names=label_encoder.classes_))
# # Save model:
# # joblib.dump(lgbm_model, os.path.join(script_dir, "career_lgbm.pkl"))

# --- CatBoost Classifier ---
# from catboost import CatBoostClassifier
# # CatBoost handles categorical features natively and is robust to overfitting.
# # It can be slower to train and increases project dependencies.
# # Common hyperparameters: iterations, learning_rate, depth, l2_leaf_reg, subsample, rsm.
# catboost_model = CatBoostClassifier(
#     iterations=300,
#     learning_rate=0.05,
#     depth=8,
#     l2_leaf_reg=3.0,
#     subsample=0.8,
#     rsm=0.8,
#     loss_function='MultiClass',
#     random_seed=42,
#     verbose=0,
#     task_type='CPU'  # Use 'GPU' if available
# )
# catboost_model.fit(X_train_scaled, y_train)
# # Evaluation:
# # print(classification_report(y_test, catboost_model.predict(X_test_scaled), target_names=label_encoder.classes_))
# # Save model:
# # joblib.dump(catboost_model, os.path.join(script_dir, "career_catboost.pkl"))

# --- Support Vector Machine (SVM) ---
# from sklearn.svm import SVC
# # SVMs can be powerful for smaller datasets and high-dimensional spaces.
# # They scale poorly with large datasets and multi-class problems.
# # Common hyperparameters: kernel, C, gamma, degree (for poly kernel), class_weight.
# svm_model = SVC(
#     kernel='rbf',  # Try 'linear', 'poly', 'sigmoid' as alternatives
#     C=1.0,         # Regularization parameter
#     gamma='scale', # Kernel coefficient
#     class_weight='balanced',
#     probability=True,  # Needed for predict_proba
#     random_state=42
# )
# svm_model.fit(X_train_scaled, y_train)
# # Evaluation:
# # print(classification_report(y_test, svm_model.predict(X_test_scaled), target_names=label_encoder.classes_))
# # Save model:
# # joblib.dump(svm_model, os.path.join(script_dir, "career_svm.pkl"))

# --- Logistic Regression ---
# from sklearn.linear_model import LogisticRegression
# # Logistic Regression is simple, interpretable, and fast.
# # It may underperform on complex, non-linear problems like this one.
# # Common hyperparameters: penalty, C, solver, max_iter, class_weight.
# logreg_model = LogisticRegression(
#     penalty='l2',         # Regularization type
#     C=1.0,               # Inverse of regularization strength
#     solver='lbfgs',      # Good for multinomial problems
#     max_iter=2000,       # Increase if not converging
#     class_weight='balanced',
#     multi_class='multinomial',
#     random_state=42,
#     n_jobs=-1
# )
# logreg_model.fit(X_train_scaled, y_train)
# # Evaluation:
# # print(classification_report(y_test, logreg_model.predict(X_test_scaled), target_names=label_encoder.classes_))
# # Save model:
# # joblib.dump(logreg_model, os.path.join(script_dir, "career_logreg.pkl"))

# --- Multi-layer Perceptron (Neural Network) ---
# from sklearn.neural_network import MLPClassifier
# # MLPs can model complex relationships, but require more data and tuning.
# # They are less interpretable and can be slow to train.
# # Common hyperparameters: hidden_layer_sizes, activation, solver, alpha, learning_rate, max_iter.
# mlp_model = MLPClassifier(
#     hidden_layer_sizes=(256, 128, 64),  # More layers/neurons for complexity
#     activation='relu',
#     solver='adam',
#     alpha=0.0001,         # L2 penalty (regularization)
#     learning_rate='adaptive',
#     max_iter=500,
#     early_stopping=True,  # Stop if validation score not improving
#     random_state=42
# )
# mlp_model.fit(X_train_scaled, y_train)
# # Evaluation:
# # print(classification_report(y_test, mlp_model.predict(X_test_scaled), target_names=label_encoder.classes_))
# # Save model:
# # joblib.dump(mlp_model, os.path.join(script_dir, "career_mlp.pkl"))
# ================= END ALTERNATIVES =================

# Model Evaluation
print("✅ Random Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test_scaled), target_names=label_encoder.classes_))

print("\n✅ XGBoost Classification Report:")
print(classification_report(y_test, xgb_model.predict(X_test_scaled), target_names=label_encoder.classes_))

# Save Models, Scaler, and Label Encoder
joblib.dump(rf_model, os.path.join(script_dir, "career_rf.pkl"))
joblib.dump(xgb_model, os.path.join(script_dir, "career_xgb.pkl"))
joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(script_dir, "label_encoder.pkl"))

print("\n✅ Models, scaler, and label encoder saved successfully.")
