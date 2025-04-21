# Tests for Career Recommender Models

This directory contains tests for the recommender models, focusing on the LightGBM implementation.

## Running Tests

To run all tests:

```bash
# From the recommender-models directory
python -m pytest tests/

# Or using the run_tests.py script
python tests/run_tests.py
```

## Test Structure

- `test_lightgbm.py`: Tests the LightGBM model functionality
- `test_train_script.py`: Tests the training script functionality
- `test_model_functionality.py`: Tests model loading and prediction format
- `test_recommender_api.py`: Tests the CareerRecommender API 
- `test_performance.py`: Tests prediction performance

## Test Types

1. **Model Loading Tests**: Tests that trained models can be loaded correctly
2. **Prediction Format Tests**: Tests that predictions return the expected format
3. **Input Validation Tests**: Tests that the model properly handles invalid inputs
4. **Known Input-Output Tests**: Tests with specific inputs that should produce known outputs
5. **Performance Tests**: Tests that predictions complete within a reasonable time

## CI/CD Integration

These tests are configured to run automatically on GitHub Actions when:
- A pull request is opened against the main branch
- Changes are pushed to the main branch

See the workflow configuration in `.github/workflows/python-tests.yml`. 