"""
recommender.py
=============
A simple API for career recommendations using the trained LightGBM model.
"""

import os
import numpy as np
import joblib

class CareerRecommender:
    """A simple career recommender based on the trained LightGBM model."""
    
    def __init__(self, model_path=None, label_encoder_path=None):
        """Initialize the recommender with model and label encoder paths."""
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.artifacts_dir = os.path.join(self.script_dir, "artifacts")
        
        # Set default paths if not provided
        if model_path is None:
            model_path = os.path.join(self.artifacts_dir, "career_lgbm.pkl")
        if label_encoder_path is None:
            label_encoder_path = os.path.join(self.artifacts_dir, "label_encoder.pkl")
        
        # Load model and label encoder
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)
    
    def predict(self, exam_scores):
        """
        Predict career based on exam scores.
        
        Parameters:
        -----------
        exam_scores : list or array-like
            A list of 7 exam scores
            
        Returns:
        --------
        dict
            A dictionary containing the predicted career, confidence,
            and probabilities for all careers
        """
        # Validate input
        if not isinstance(exam_scores, (list, np.ndarray)) or len(exam_scores) != 7:
            raise ValueError("Exam scores must be a list or array of 7 values")
        
        # Convert to numpy array for prediction
        input_array = np.array([exam_scores])
        
        # Make prediction
        pred_idx = self.model.predict(input_array)[0]
        pred_career = self.label_encoder.inverse_transform([pred_idx])[0]
        
        # Get probabilities
        proba = self.model.predict_proba(input_array)[0]
        
        # Return result
        return {
            "predicted_career": pred_career,
            "confidence": float(proba[pred_idx]),
            "all_probabilities": {
                career: float(prob) 
                for career, prob in zip(self.label_encoder.classes_, proba)
            }
        }
    
    def recommend(self, math_score, physics_score, chemistry_score, 
                  biology_score, english_score, programming_score, 
                  economics_score):
        """
        A more user-friendly interface for the predict method.
        
        Parameters:
        -----------
        math_score, physics_score, chemistry_score, biology_score,
        english_score, programming_score, economics_score : float
            Individual exam scores (typically 0-4 range)
            
        Returns:
        --------
        dict
            Same as predict() method
        """
        exam_scores = [
            math_score, physics_score, chemistry_score, biology_score,
            english_score, programming_score, economics_score
        ]
        return self.predict(exam_scores)


def get_recommender():
    """Helper function to get a recommender instance."""
    return CareerRecommender()


if __name__ == "__main__":
    # Example usage
    recommender = get_recommender()
    result = recommender.predict([3.8, 3.9, 3.7, 3.6, 3.8, 3.9, 3.7])
    print(f"Predicted career: {result['predicted_career']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nAll career probabilities:")
    for career, prob in result['all_probabilities'].items():
        print(f"  {career}: {prob:.2f}")
