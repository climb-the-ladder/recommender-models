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
df = pd.read_csv(csv_path)

# Rename columns consistently (so it matches our frontend and backend)
df = df.rename(columns={
    "Extracurricular_Activities": "Extracurriculars",
    "Field_Specific_Courses": "Courses",
    "Industry_Certifications": "Certifications",
    "Internships": "InternshipExperience",
    "Analytical_Skills": "AnalyticalSkills"
})

# Convert categorical Yes/No columns to binary - required by xgboost, dont remove
categorical_columns = ["Extracurriculars", "InternshipExperience", "Courses", "Certifications"]
for col in categorical_columns:
    df[col] = df[col].fillna("No").map({"Yes": 1, "No": 0})

df.fillna(0, inplace=True)

# Features and Target x features y target
X = df.drop(columns=["Career"])
y = df["Career"]

# Encode categorical target labels FIRST - important
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# apply SMOTE after encoding (might remove later)
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

# Train-Test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200, 
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_scaled, y_train)

# XGBoost model
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

# to evaluate models
print("Random Forest Classification Report:")
print(classification_report(y_test, rf_model.predict(X_test_scaled), target_names=label_encoder.classes_))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, xgb_model.predict(X_test_scaled), target_names=label_encoder.classes_))

joblib.dump(rf_model, os.path.join(script_dir, "career_rf.pkl"))
joblib.dump(xgb_model, os.path.join(script_dir, "career_xgb.pkl"))
joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))
joblib.dump(label_encoder, os.path.join(script_dir, "label_encoder.pkl"))

print("✅ Models trained and saved successfully.")
