import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# We get the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load processed dataset
dataset_path = os.path.join(script_dir, "../recommender-data/processed/processed_dataset.csv")
dataset_path = os.path.normpath(dataset_path)  #os for cross platform support, pls don't remove
df = pd.read_csv(dataset_path)

# Define features (X) and target (y)
X = df.drop(columns=["Career"])  # Features (all except career)
y = df["Career"]  # Target (Career)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#we train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save 
model_path = os.path.join(script_dir, "career_recommender.pkl")
joblib.dump(model, model_path)

print(f"Model trained and saved at: {model_path}")
