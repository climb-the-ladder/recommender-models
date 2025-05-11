import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add parent directory to path to allow importing from train_lightgbm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightgbm import LGBMClassifier

class TestLightGBM(unittest.TestCase):
    
    def test_model_imports(self):
        """Test that all required imports work."""
        try:
            import lightgbm as lgb
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            from sklearn.metrics import classification_report, accuracy_score, f1_score
            from imblearn.over_sampling import SMOTE
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Import error: {e}")
    
    def test_dataset_exists(self):
        """Test that the dataset file exists."""
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(script_dir, "processed_dataset.csv")
        self.assertTrue(os.path.exists(csv_path), f"Dataset file not found at {csv_path}")
    
    def test_simple_model_creation(self):
        """Test that we can create a LightGBM model."""
        model = LGBMClassifier(
            objective="multiclass",
            num_class=5,  # Assuming 5 classes for testing
            n_estimators=10,  # Small value for faster tests
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        self.assertIsNotNone(model, "Failed to create LightGBM model")
        
    def test_simple_model_fit(self):
        """Test that we can fit a simple model on random data."""
        # Create some dummy data
        X = np.random.rand(100, 7)  # 7 features based on your dataset
        y = np.random.randint(0, 5, 100)  # 5 classes
        
        # Create and fit model
        model = LGBMClassifier(
            objective="multiclass",
            num_class=5,
            n_estimators=10,  # Small value for faster tests
            learning_rate=0.1,
            num_leaves=31,
            random_state=42
        )
        model.fit(X, y)
        
        # Make a prediction
        y_pred = model.predict(X[:1])
        self.assertIsNotNone(y_pred, "Model prediction failed")
        self.assertEqual(len(y_pred), 1, "Prediction length mismatch")

if __name__ == "__main__":
    unittest.main() 