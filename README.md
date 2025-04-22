# Career Recommender Models Repository

This repository contains the machine learning models and analysis tools for the Career Recommendation System.

## Repository Structure


## Models

### XGBoost Career Prediction Model

The main model (`career_xgb.pkl`) is an XGBoost classifier trained to predict suitable career paths based on a student's academic performance across different subjects. The model takes subject scores as input and outputs a recommended career field.

Key features:
- Trained on processed student academic data
- Uses 7 subject scores as features (math, history, physics, chemistry, biology, english, geography)
- Predicts from multiple career categories including Software Engineer, Doctor, Lawyer, etc.
- Achieves high accuracy in matching students to appropriate career paths

### Supporting Models

1. **scaler.pkl**: StandardScaler that normalizes input features to improve model performance
2. **label_encoder.pkl**: LabelEncoder that converts between career field names and their numerical representations

## Data Insights

For a detailed analysis of the data, run the insights generation script:

```bash
cd recommender-models
python insights.py
```

This will generate a new folder called `recommender-insights` with a comprehensive PDF report with visualizations and key insights about:
- Career distributions
- Subject correlations
- Career-subject relationships
- Key insights and recommendations

The report is saved to `recommender-insights/reports/career_insights_report.pdf`.

## Training the Model

To retrain the model with updated data:

```bash
python train_model.py
```

The training script:
1. Loads the processed dataset
2. Preprocesses the data (scaling, encoding)
3. Splits the data into training and testing sets
4. Trains an XGBoost classifier
5. Evaluates model performance
6. Saves the model and preprocessing components

## Model Usage

The model is used by the AI service component of the Career Recommendation System. It can be loaded and used as follows:

```python
import joblib
import pandas as pd

# Load the model and preprocessing components
model = joblib.load('career_xgb.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Example input data
input_data = {
    "math_score": 92,
    "history_score": 75,
    "physics_score": 94,
    "chemistry_score": 96,
    "biology_score": 89,
    "english_score": 80,
    "geography_score": 78
}

# Prepare the input
features = pd.DataFrame([input_data])
features_scaled = scaler.transform(features)

# Make prediction
predicted_label = model.predict(features_scaled)[0]
predicted_career = label_encoder.inverse_transform([predicted_label])[0]

print(f"Recommended Career: {predicted_career}")
```

## Model Performance

The XGBoost model achieves:
- High accuracy in predicting appropriate career paths
- Good generalization to new student data
- Balanced precision and recall across different career categories

## Dependencies

- scikit-learn: For data preprocessing and model evaluation
- xgboost: For the gradient boosting model
- pandas: For data manipulation
- numpy: For numerical operations
- matplotlib & seaborn: For data visualization
- joblib: For model serialization

## Integration

This model is integrated with:
1. The AI service component (`recommender-ai/app.py`)
2. The backend API (`recommender-backend/routes/recommendations.py`)
3. The data insights module (`insights.py`)

## License

This model is for educational and demonstration purposes only. Do not use for commercial applications without proper authorization.


