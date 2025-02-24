import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

script_dir = os.path.dirname(os.path.abspath(__file__))

dataset_path = os.path.join(script_dir, "../recommender-data/processed/processed_dataset.csv")
df = pd.read_csv(dataset_path)

df = df.rename(columns={
    "Extracurricular_Activities": "Extracurriculars",
    "Field_Specific_Courses": "Courses",
    "Industry_Certifications": "Certifications",
    "Internships": "InternshipExperience",
    "Analytical_Skills": "AnalyticalSkills"
})

missing_cols = df.columns[df.isnull().any()].tolist()
print(f"⚠️ Missing values detected in: {missing_cols}")

#Fill missing values before converting Yes/No to 1/0
categorical_columns = ["Extracurriculars", "InternshipExperience", "Courses", "Certifications"]
for col in categorical_columns:
    df[col] = df[col].fillna("No")  # Assume missing values mean "No"
    df[col] = df[col].map({"Yes": 1, "No": 0})  # Convert to integers

df.fillna(0, inplace=True)

X = df.drop(columns=["Career"])  # Features
y = df["Career"]  # Target

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Increase weight for technical features to prioritize them
feature_weights = {
    "GPA": 1.0,  
    "Coding_Skills": 1.5, 
    "Problem_Solving_Skills": 1.5,
    "AnalyticalSkills": 1.5,
    "InternshipExperience": 1.3,
    "Research_Experience": 1.2,
    "Projects": 1.2
}

# Multiply important feature columns by their respective weights
for feature, weight in feature_weights.items():
    if feature in X_balanced.columns:
        X_balanced[feature] = X_balanced[feature] * weight

# the split for training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# the Random Forest model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train_scaled, y_train)

# Save model & scaler
joblib.dump(model, os.path.join(script_dir, "career_recommender.pkl"))
joblib.dump(scaler, os.path.join(script_dir, "scaler.pkl"))

print(f"✅ Model trained with balanced data and saved!")
