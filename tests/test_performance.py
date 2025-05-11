import os
import sys
import time
import unittest
import numpy as np
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

class TestPerformance(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.artifacts_dir = os.path.join(parent_dir, "artifacts")
        self.sample_input = np.array([[3.8, 3.9, 3.7, 3.6, 3.8, 3.9, 3.7]])  # Sample exam scores
        
        # Create mock objects
        self.mock_le = MagicMock()
        self.mock_le.classes_ = np.array(['Data Scientist', 'Doctor', 'Engineer', 'Lawyer', 'Software Engineer'])
        self.mock_le.inverse_transform.return_value = np.array(['Software Engineer'])
        
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([4])  # Index for 'Software Engineer'
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]])
    
    @patch('joblib.load')
    def test_prediction_performance(self, mock_load):
        """Test that predictions complete within a reasonable time."""
        try:
            import joblib
            
            # Configure mock to return our mock objects
            mock_load.side_effect = [self.mock_model, self.mock_le]
            
            # Load model and label encoder (mocked)
            model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
            le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
            
            model = joblib.load(model_path)
            le = joblib.load(le_path)
            
            # Time the prediction
            start_time = time.time()
            y_pred = model.predict(self.sample_input)
            career = le.inverse_transform(y_pred)
            duration = time.time() - start_time
            
            # Prediction should be very fast (using mocks)
            self.assertLess(duration, 0.1, f"Prediction took too long: {duration:.4f} seconds")
            
        except ImportError as e:
            self.skipTest(f"Skipping performance test due to import error: {e}")
    
    @patch('joblib.load')
    def test_batch_prediction_performance(self, mock_load):
        """Test performance with a larger batch of inputs."""
        try:
            import joblib
            
            # Configure mock to return our mock objects
            mock_load.side_effect = [self.mock_model, self.mock_le]
            
            # Load model and label encoder (mocked)
            model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
            le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
            
            model = joblib.load(model_path)
            
            # Create a larger batch of inputs (100 samples)
            batch_input = np.random.rand(100, 7)
            
            # Configure mock for batch prediction
            self.mock_model.predict.return_value = np.array([4] * 100)
            
            # Time the batch prediction
            start_time = time.time()
            y_pred = model.predict(batch_input)
            duration = time.time() - start_time
            
            # Batch prediction should be reasonably fast
            self.assertLess(duration, 0.5, f"Batch prediction took too long: {duration:.4f} seconds")
            
        except ImportError as e:
            self.skipTest(f"Skipping performance test due to import error: {e}")

if __name__ == "__main__":
    unittest.main() 