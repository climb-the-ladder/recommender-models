import pandas as pd
import os
import sys
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib
from xgboost import XGBClassifier
from typing import Tuple, Dict, Any, Union, TypedDict
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelOutput(TypedDict):
    """Type definition for the model training output dictionary."""
    rf_model: RandomForestClassifier
    xgb_model: XGBClassifier
    scaler: StandardScaler
    label_encoder: LabelEncoder
    X_test: np.ndarray
    y_test: np.ndarray
    label_classes: np.ndarray

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains both Random Forest and XGBoost models on the given dataset with preprocessing steps.

    Parameters:
    -----------
    X : pd.DataFrame
        The feature matrix containing all predictor variables.
    y : pd.Series
        The target variable (career aspirations).
    test_size : float, default=0.2
        The proportion of the dataset to include in the test split.
    random_state : int, default=42
        Random state for reproducibility.

    Returns:
    --------
    Dict[str, Any]
        A dictionary containing:
        - 'rf_model': Trained Random Forest model
        - 'xgb_model': Trained XGBoost model
        - 'scaler': Fitted StandardScaler
        - 'label_encoder': Fitted LabelEncoder
        - 'X_test': Test features
        - 'y_test': Test labels
        - 'label_classes': Original label classes
    """
    # Encode target labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Apply SMOTE for balancing
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X, y_encoded)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=test_size, random_state=random_state
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        random_state=random_state
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
        random_state=random_state
    )
    xgb_model.fit(X_train_scaled, y_train)

    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'X_test': X_test_scaled,
        'y_test': y_test,
        'label_classes': label_encoder.classes_
    }

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
    logger.info(f"Checking for data file at: {path}")
    if os.path.exists(path):
        csv_path = path
        logger.info(f"Found data file at: {csv_path}")
        break

if csv_path is None:
    logger.error("Could not find the processed dataset file.")
    # List directories to help debug
    logger.info("Contents of base directory:")
    for item in os.listdir(base_dir):
        logger.info(f"  - {item}")
    
    # Check if we can find the recommender-data directory
    data_dir = os.path.join(base_dir, "recommender-data")
    if os.path.exists(data_dir):
        logger.info(f"Contents of {data_dir}:")
        for item in os.listdir(data_dir):
            logger.info(f"  - {item}")
        
        # Check if the processed directory exists
        processed_dir = os.path.join(data_dir, "processed")
        if os.path.exists(processed_dir):
            logger.info(f"Contents of {processed_dir}:")
            for item in os.listdir(processed_dir):
                logger.info(f"  - {item}")
    
    sys.exit(1)

try:
    # Load the preprocessed dataset
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded dataframe with shape: {df.shape}")

    # Features and Target
    X = df.drop(columns=["career_aspiration"])
    y = df["career_aspiration"]

    # Train models
    models = train_model(X, y)

    # Model Evaluation
    logger.info("Random Forest Classification Report:")
    logger.info("\n" + classification_report(
        models['y_test'],
        models['rf_model'].predict(models['X_test']),
        target_names=models['label_classes']
    ))

    logger.info("XGBoost Classification Report:")
    logger.info("\n" + classification_report(
        models['y_test'],
        models['xgb_model'].predict(models['X_test']),
        target_names=models['label_classes']
    ))

    # Save Models, Scaler, and Label Encoder
    joblib.dump(models['rf_model'], os.path.join(script_dir, "career_rf.pkl"))
    joblib.dump(models['xgb_model'], os.path.join(script_dir, "career_xgb.pkl"))
    joblib.dump(models['scaler'], os.path.join(script_dir, "scaler.pkl"))
    joblib.dump(models['label_encoder'], os.path.join(script_dir, "label_encoder.pkl"))

    logger.info("Models, scaler, and label encoder saved successfully.")
    logger.info("train_model.py executed successfully.")
    
except Exception as e:
    logger.error(f"Error during model training: {str(e)}")
    sys.exit(1)
