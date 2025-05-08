"""
Constants used throughout the recommender models project.
"""

# File paths and names
PROCESSED_DATASET_FILENAME = "processed_dataset.csv"
CAREER_RF_MODEL_FILENAME = "career_rf.pkl"
CAREER_XGB_MODEL_FILENAME = "career_xgb.pkl"
SCALER_FILENAME = "scaler.pkl"
LABEL_ENCODER_FILENAME = "label_encoder.pkl"

# Data paths
DATA_PATHS = [
    "recommender-data-main/processed/processed_dataset.csv",  # Local development path
    "recommender-data/processed/processed_dataset.csv",       # Docker container path
    "../recommender-data/processed/processed_dataset.csv",    # Relative path
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Random Forest parameters
RF_N_ESTIMATORS = 200
RF_CLASS_WEIGHT = 'balanced'

# XGBoost parameters
XGB_N_ESTIMATORS = 300
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 8
XGB_OBJECTIVE = 'multi:softmax'
XGB_EVAL_METRIC = 'mlogloss'

# Target column name
TARGET_COLUMN = "career_aspiration" 