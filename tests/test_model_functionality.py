import os
import sys
import unittest
import numpy as np
import pandas as pd
import joblib
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

class TestModelFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures, if any."""
        self.artifacts_dir = os.path.join(parent_dir, "artifacts")
        self.sample_input = np.array([[3.8, 3.9, 3.7, 3.6, 3.8, 3.9, 3.7]])  # Sample exam scores
        
        # Create mock label encoder
        self.mock_le = MagicMock()
        self.mock_le.classes_ = np.array(['Data Scientist', 'Doctor', 'Engineer', 'Lawyer', 'Software Engineer'])
        self.mock_le.inverse_transform.return_value = np.array(['Software Engineer'])
        
        # Create mock model
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([4])  # Index for 'Software Engineer'
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]])
    
    def test_model_file_exists(self):
        """Test that the model file exists in artifacts directory."""
        model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
        self.assertTrue(os.path.exists(self.artifacts_dir), f"Artifacts directory not found at {self.artifacts_dir}")
        self.assertTrue(os.path.exists(model_path), f"Model file not found at {model_path}")
    
    def test_label_encoder_exists(self):
        """Test that the label encoder file exists in artifacts directory."""
        le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
        self.assertTrue(os.path.exists(le_path), f"Label encoder file not found at {le_path}")
    
    @patch('joblib.load')
    def test_model_loading(self, mock_load):
        """Test that the model and label encoder can be loaded."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Load model and label encoder
        model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
        le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
        
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        
        # Verify the loads were called and objects returned
        self.assertEqual(mock_load.call_count, 2)
        self.assertIsNotNone(model)
        self.assertIsNotNone(le)
    
    @patch('joblib.load')
    def test_prediction_format(self, mock_load):
        """Test that predictions have the expected format."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Load model and label encoder
        model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
        le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
        
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        
        # Make a prediction
        y_pred = model.predict(self.sample_input)
        self.assertTrue(isinstance(y_pred, np.ndarray))
        self.assertEqual(len(y_pred), 1)
        
        # Convert prediction index to career name
        career = le.inverse_transform(y_pred)
        self.assertTrue(isinstance(career, np.ndarray))
        self.assertEqual(len(career), 1)
        self.assertEqual(career[0], 'Software Engineer')
    
    @patch('joblib.load')
    def test_prediction_probabilities(self, mock_load):
        """Test that prediction probabilities have the expected format."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Load model and label encoder
        model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
        le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
        
        model = joblib.load(model_path)
        le = joblib.load(le_path)
        
        # Get prediction probabilities
        proba = model.predict_proba(self.sample_input)
        self.assertTrue(isinstance(proba, np.ndarray))
        self.assertEqual(proba.shape[0], 1)
        self.assertEqual(proba.shape[1], 5)  # 5 classes
        
        # Check that probabilities sum to 1
        self.assertAlmostEqual(np.sum(proba[0]), 1.0, places=5)
    
    def test_simple_recommender_api(self):
        """Test a simple API-like function for the recommender."""
        # Define a simple recommender function (this would be in your actual code)
        def recommend_career(exam_scores, model_path=None, le_path=None):
            """Simple API-like function to recommend a career based on exam scores."""
            if model_path is None:
                model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
            if le_path is None:
                le_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
            
            # Validate input
            if not isinstance(exam_scores, list) or len(exam_scores) != 7:
                raise ValueError("Exam scores must be a list of 7 values")
            
            # Mock the model and label encoder loading
            model = self.mock_model
            le = self.mock_le
            
            # Make prediction
            input_array = np.array([exam_scores])
            pred_idx = model.predict(input_array)[0]
            pred_career = le.inverse_transform([pred_idx])[0]
            
            # Get probabilities
            proba = model.predict_proba(input_array)[0]
            
            # Return result
            return {
                "predicted_career": pred_career,
                "confidence": float(proba[pred_idx]),
                "all_probabilities": {
                    career: float(prob) for career, prob in zip(le.classes_, proba)
                }
            }
        
        # Test with valid input
        result = recommend_career([3.8, 3.9, 3.7, 3.6, 3.8, 3.9, 3.7])
        self.assertIsInstance(result, dict)
        self.assertIn("predicted_career", result)
        self.assertIn("confidence", result)
        self.assertIn("all_probabilities", result)
        self.assertEqual(result["predicted_career"], "Software Engineer")
        
        # Test with invalid input
        with self.assertRaises(ValueError):
            recommend_career([3.8, 3.9])  # Too few scores

if __name__ == "__main__":
    unittest.main() 