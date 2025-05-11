import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

import numpy as np
from recommender import CareerRecommender

class TestRecommenderAPI(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock objects
        self.mock_le = MagicMock()
        self.mock_le.classes_ = np.array(['Data Scientist', 'Doctor', 'Engineer', 'Lawyer', 'Software Engineer'])
        self.mock_le.inverse_transform.return_value = np.array(['Software Engineer'])
        
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.array([4])  # Index for 'Software Engineer'
        self.mock_model.predict_proba.return_value = np.array([[0.1, 0.1, 0.1, 0.1, 0.6]])
    
    @patch('joblib.load')
    def test_recommender_initialization(self, mock_load):
        """Test that the recommender can be initialized."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Initialize recommender
        recommender = CareerRecommender()
        
        # Verify that joblib.load was called twice
        self.assertEqual(mock_load.call_count, 2)
        
        # Verify that model and label encoder were loaded
        self.assertIsNotNone(recommender.model)
        self.assertIsNotNone(recommender.label_encoder)
    
    @patch('joblib.load')
    def test_predict_method(self, mock_load):
        """Test the predict method of the recommender."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Initialize recommender
        recommender = CareerRecommender()
        
        # Call predict method
        result = recommender.predict([3.8, 3.9, 3.7, 3.6, 3.8, 3.9, 3.7])
        
        # Verify result format
        self.assertIsInstance(result, dict)
        self.assertIn('predicted_career', result)
        self.assertIn('confidence', result)
        self.assertIn('all_probabilities', result)
        
        # Verify result values
        self.assertEqual(result['predicted_career'], 'Software Engineer')
        self.assertAlmostEqual(result['confidence'], 0.6)
        self.assertEqual(len(result['all_probabilities']), 5)
    
    @patch('joblib.load')
    def test_recommend_method(self, mock_load):
        """Test the recommend method of the recommender."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Initialize recommender
        recommender = CareerRecommender()
        
        # Call recommend method with individual scores
        result = recommender.recommend(
            math_score=3.8,
            physics_score=3.9,
            chemistry_score=3.7,
            biology_score=3.6,
            english_score=3.8,
            programming_score=3.9,
            economics_score=3.7
        )
        
        # Verify result format
        self.assertIsInstance(result, dict)
        self.assertIn('predicted_career', result)
        self.assertIn('confidence', result)
        self.assertIn('all_probabilities', result)
        
        # Verify result values
        self.assertEqual(result['predicted_career'], 'Software Engineer')
        self.assertAlmostEqual(result['confidence'], 0.6)
        self.assertEqual(len(result['all_probabilities']), 5)
    
    @patch('joblib.load')
    def test_invalid_input(self, mock_load):
        """Test that invalid inputs raise appropriate exceptions."""
        # Configure mock to return our mock objects
        mock_load.side_effect = [self.mock_model, self.mock_le]
        
        # Initialize recommender
        recommender = CareerRecommender()
        
        # Test with too few scores
        with self.assertRaises(ValueError):
            recommender.predict([3.8, 3.9])
        
        # Test with too many scores
        with self.assertRaises(ValueError):
            recommender.predict([3.8, 3.9, 3.7, 3.6, 3.8, 3.9, 3.7, 4.0])
        
        # Test with non-list input
        with self.assertRaises(ValueError):
            recommender.predict("not a list")
    
    @patch('joblib.load')
    def test_known_input_output(self, mock_load):
        """Test with specific inputs that should produce known outputs."""
        # For this test, we'll configure the mocks to return different values for different inputs
        
        # Create a more complex mock model that returns different outputs based on inputs
        def mock_predict(X):
            # Example logic: High programming score = Software Engineer
            if X[0][5] >= 3.8:  # Index 5 is programming_score
                return np.array([4])  # 'Software Engineer'
            # High biology score = Doctor
            elif X[0][3] >= 3.8:  # Index 3 is biology_score
                return np.array([1])  # 'Doctor'
            # Default to Data Scientist
            else:
                return np.array([0])  # 'Data Scientist'
        
        def mock_inverse_transform(indices):
            mapping = {
                0: 'Data Scientist',
                1: 'Doctor',
                2: 'Engineer',
                3: 'Lawyer',
                4: 'Software Engineer'
            }
            return np.array([mapping[idx] for idx in indices])
        
        # Configure mocks
        complex_model = MagicMock()
        complex_model.predict = mock_predict
        complex_model.predict_proba.return_value = np.array([[0.2, 0.2, 0.2, 0.2, 0.2]])
        
        complex_le = MagicMock()
        complex_le.classes_ = np.array(['Data Scientist', 'Doctor', 'Engineer', 'Lawyer', 'Software Engineer'])
        complex_le.inverse_transform = mock_inverse_transform
        
        mock_load.side_effect = [complex_model, complex_le]
        
        # Initialize recommender
        recommender = CareerRecommender()
        
        # Test case for Software Engineer (high programming score)
        se_input = [3.5, 3.5, 3.5, 3.5, 3.5, 3.9, 3.5]  # High programming score
        result = recommender.predict(se_input)
        self.assertEqual(result['predicted_career'], 'Software Engineer')
        
        # Reset mocks for next test
        mock_load.reset_mock()
        mock_load.side_effect = [complex_model, complex_le]
        
        # Test case for Doctor (high biology score)
        doctor_input = [3.5, 3.5, 3.5, 3.9, 3.5, 3.5, 3.5]  # High biology score
        recommender = CareerRecommender()
        result = recommender.predict(doctor_input)
        self.assertEqual(result['predicted_career'], 'Doctor')
        
        # Reset mocks for next test
        mock_load.reset_mock()
        mock_load.side_effect = [complex_model, complex_le]
        
        # Test case for Data Scientist (default)
        ds_input = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5]  # Average scores
        recommender = CareerRecommender()
        result = recommender.predict(ds_input)
        self.assertEqual(result['predicted_career'], 'Data Scientist')

if __name__ == "__main__":
    unittest.main() 