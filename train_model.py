import pandas as pd
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier

# Set up paths - look for data in multiple possible locations
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)

# Define multiple possible paths for the data file
possible_paths = [
    os.path.join(base_dir, "recommender-data-main/processed/processed_dataset.csv"),  # Local development path
    os.path.join(base_dir, "recommender-data/processed/processed_dataset.csv"),       # Docker container path
    os.path.join(script_dir, "../recommender-data/processed/processed_dataset.csv"),  # Relative path
]

# Try to find the data file
csv_path = None
for path in possible_paths:
    print(f"Checking for data file at: {path}")
    if os.path.exists(path):
        csv_path = path
        print(f"✅ Found data file at: {csv_path}")
        break

if csv_path is None:
    print("❌ Error: Could not find the processed dataset file.")
    # List directories to help debug
    print("Contents of base directory:")
    for item in os.listdir(base_dir):
        print(f"  - {item}")
    
    # Check if we can find the recommender-data directory
    data_dir = os.path.join(base_dir, "recommender-data")
    if os.path.exists(data_dir):
        print(f"Contents of {data_dir}:")
        for item in os.listdir(data_dir):
            print(f"  - {item}")
        
        # Check if the processed directory exists
        processed_dir = os.path.join(data_dir, "processed")
        if os.path.exists(processed_dir):
            print(f"Contents of {processed_dir}:")
            for item in os.listdir(processed_dir):
                print(f"  - {item}")
    
    sys.exit(1)

try:
    # Load the preprocessed dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded dataframe with shape: {df.shape}")

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
    
except Exception as e:
    print(f"❌ Error during model training: {str(e)}")
    sys.exit(1)
