# ML Engine Usage Guide

## Overview

The Magik Merlin ML Platform features a custom-built ML engine designed to replace PyCaret with a modern, lightweight, and extensible solution. The engine provides sklearn-compatible interfaces for popular gradient boosting libraries and deep learning frameworks.

## Why Custom ML Engine?

We built our own ML engine instead of using PyCaret for several key reasons:

1. **No Version Lock-in**: PyCaret 3.3.2 only supports Python up to 3.11, limiting us to older Python versions
2. **Future-Proof**: Built for Python 3.13+ with modern language features
3. **Lightweight**: No bloated dependencies - only the libraries you need
4. **Full Control**: Complete customization of ML workflows and interfaces
5. **Better Maintainability**: Clean, documented codebase we fully control

## Supported Libraries

### Gradient Boosting Models
- ✅ **XGBoost** 2.0+ - Extreme Gradient Boosting
- ✅ **LightGBM** 4.0+ - Microsoft's gradient boosting framework
- ✅ **CatBoost** 1.2+ - Yandex's categorical boosting

### Deep Learning (Coming Soon)
- 🚧 **PyTorch** 2.0+ - Deep learning framework
- 🚧 **Lightning** 2.0+ - PyTorch Lightning for simplified training

### Hyperparameter Optimization
- ✅ **Optuna** 3.0+ - Automatic hyperparameter tuning

## Quick Start

### Installation

```bash
# Install ML engine dependencies
uv sync --extra ml

# This installs: xgboost, lightgbm, catboost, torch, lightning, optuna, shap
```

### Basic Classification Example

```python
from src.core.ml_engine import AutoMLPipeline
import pandas as pd

# Load your data
data = pd.read_csv("your_data.csv")
X = data.drop("target", axis=1)
y = data["target"]

# Create AutoML pipeline
pipeline = AutoMLPipeline(task_type="classification")

# Compare all available models with cross-validation
results = pipeline.compare_models(X, y, cv=5, test_size=0.2)

print(results)
#      model  cv_mean  cv_std  test_score
# 0  xgboost  0.92     0.03    0.94
# 1  lightgbm 0.91     0.02    0.93
# 2  catboost 0.90     0.04    0.92

# Get the best model
best_model = pipeline.get_best_model()

# Make predictions
predictions = pipeline.predict(X_new)
```

### Basic Regression Example

```python
# Same interface for regression
pipeline = AutoMLPipeline(task_type="regression")

results = pipeline.compare_models(X, y, cv=5)

# Get R² scores
print(f"Best model: {pipeline.best_model_name}")
print(f"Best R² score: {results.iloc[0]['test_score']}")
```

## Advanced Usage

### Hyperparameter Optimization

```python
from src.core.ml_engine import AutoMLPipeline

pipeline = AutoMLPipeline(task_type="classification")

# First, compare models to find the best one
results = pipeline.compare_models(X, y)

# Then optimize the best model's hyperparameters
optimization_result = pipeline.optimize_hyperparameters(
    X, y,
    model_name=pipeline.best_model_name,
    n_trials=100,  # Number of Optuna trials
    cv=5           # Cross-validation folds
)

print(f"Best parameters: {optimization_result['best_params']}")
print(f"Best score: {optimization_result['best_score']}")

# Use the optimized model
optimized_model = optimization_result['model']
predictions = optimized_model.predict(X_new)
```

### Using Individual Models

```python
from src.core.ml_engine import XGBoostClassifier, LightGBMRegressor

# XGBoost classifier with custom parameters
xgb_model = XGBoostClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

xgb_model.fit(X_train, y_train)
predictions = xgb_model.predict(X_test)
probabilities = xgb_model.predict_proba(X_test)

# Get feature importance
importance_df = xgb_model.get_feature_importance()
print(importance_df.head())
#    feature  importance
# 0  age      0.25
# 1  income   0.18
# 2  credit   0.15
```

### MLflow Integration

```python
from src.core.ml_engine import AutoMLPipeline
from src.core.experiments.tracking import ExperimentTracker

# Create experiment tracker
tracker = ExperimentTracker(
    tracking_uri="http://localhost:5000",
    experiment_name="my_experiment"
)

# Create pipeline with tracking
pipeline = AutoMLPipeline(
    task_type="classification",
    experiment_tracker=tracker
)

# All model training will be logged to MLflow
results = pipeline.compare_models(X, y)

# Parameters, metrics, and models are automatically logged!
```

## Model Registry

The ML engine includes a model registry for easy model discovery:

```python
from src.core.ml_engine import model_registry

# List all available models
all_models = model_registry.list_models()
print(all_models)
# ['xgboost_classifier', 'xgboost_regressor', 'lightgbm_classifier', ...]

# List models by category
classifiers = model_registry.list_models(category="classification")
regressors = model_registry.list_models(category="regression")

# Get a specific model class
XGBClassifier = model_registry.get_model("xgboost_classifier")
model = XGBClassifier(n_estimators=100)
```

## Sklearn Compatibility

All models are fully sklearn-compatible:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.core.ml_engine import XGBoostClassifier

# Use in sklearn pipelines
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBoostClassifier(n_estimators=100))
])

pipeline.fit(X_train, y_train)
score = pipeline.score(X_test, y_test)

# Compatible with cross_val_score, GridSearchCV, etc.
from sklearn.model_selection import cross_val_score

scores = cross_val_score(
    XGBoostClassifier(),
    X, y,
    cv=5,
    scoring='accuracy'
)
```

## Feature Importance

All gradient boosting models support feature importance extraction:

```python
model = pipeline.get_best_model()

# Get feature importance as DataFrame
importance_df = model.get_feature_importance()

# Plot with your favorite library
import matplotlib.pyplot as plt

importance_df.head(10).plot(
    x='feature',
    y='importance',
    kind='barh',
    title='Top 10 Most Important Features'
)
plt.show()
```

## Handling Categorical Features

CatBoost automatically detects and handles categorical features:

```python
from src.core.ml_engine import CatBoostClassifier

# DataFrame with mixed types
X = pd.DataFrame({
    'numeric_feat': [1.0, 2.0, 3.0],
    'category_feat': ['A', 'B', 'A'],  # Will be auto-detected
    'another_num': [10, 20, 30]
})

# CatBoost handles categorical features automatically
model = CatBoostClassifier(iterations=100)
model.fit(X, y)  # Categorical features detected and handled internally
```

For XGBoost and LightGBM, you'll need to encode categorical features manually or use pandas categorical dtype.

## Error Handling

The ML engine provides informative error messages:

```python
from src.core.ml_engine import AutoMLPipeline

pipeline = AutoMLPipeline(task_type="classification")

# Trying to predict before training
try:
    pipeline.predict(X)
except RuntimeError as e:
    print(e)  # "No models have been trained yet. Run compare_models() first."

# Invalid model name
try:
    model_registry.get_model("invalid_model")
except KeyError as e:
    print(e)  # "Model 'invalid_model' not found in registry..."
```

## Performance Tips

1. **Use Cross-Validation Wisely**: More folds = more accurate estimates but slower training
   ```python
   # Fast for prototyping
   pipeline.compare_models(X, y, cv=3)

   # More robust for final evaluation
   pipeline.compare_models(X, y, cv=10)
   ```

2. **Limit Hyperparameter Search for Speed**:
   ```python
   # Quick search
   pipeline.optimize_hyperparameters(X, y, model_name="xgboost_classifier", n_trials=20)

   # Thorough search
   pipeline.optimize_hyperparameters(X, y, model_name="xgboost_classifier", n_trials=200)
   ```

3. **Use Specific Models for Control**:
   ```python
   # AutoML is great for exploration
   pipeline.compare_models(X, y)

   # But use specific models when you know what you want
   model = XGBoostClassifier(n_estimators=1000, early_stopping_rounds=10)
   ```

## API Reference

### AutoMLPipeline

**Parameters:**
- `task_type`: str - "classification" or "regression"
- `experiment_tracker`: ExperimentTracker | None - Optional MLflow tracker
- `random_state`: int - Random seed (default: 42)

**Methods:**
- `compare_models(X, y, models=None, cv=5, test_size=0.2)` - Compare multiple models
- `optimize_hyperparameters(X, y, model_name, n_trials=50, cv=5)` - Tune hyperparameters
- `get_best_model()` - Get best performing model
- `predict(X, use_best=True)` - Make predictions
- `get_results_summary()` - Get summary of all results

### Base Model Classes

All models inherit from `BaseClassifier` or `BaseRegressor` and support:

**Methods:**
- `fit(X, y)` - Train the model
- `predict(X)` - Make predictions
- `predict_proba(X)` - Predict class probabilities (classifiers only)
- `score(X, y)` - Calculate accuracy (classification) or R² (regression)
- `get_params()` - Get model hyperparameters
- `set_params(**params)` - Set model hyperparameters
- `get_metadata()` - Get model metadata (features, is_fitted, etc.)
- `get_feature_importance()` - Get feature importance DataFrame

## Backward Compatibility

The old PyCaret interface is still available through a compatibility layer:

```python
from src.core.ml.pycaret_integration import AutoMLWorkflow

# Works like old PyCaret interface but uses new ML engine internally
workflow = AutoMLWorkflow()

results = workflow.run_automl(
    data=data,
    task_type="classification",
    target="target_column",
    tune_hyperparameters=True
)

predictions = workflow.get_predictions(data=new_data)
importance = workflow.get_model_interpretation()
```

## Troubleshooting

**Issue: "No models available for task type"**
- Solution: Install ML libraries with `uv sync --extra ml`

**Issue: "Optuna is not installed"**
- Solution: Optuna is included in ML extras: `uv sync --extra ml`

**Issue: Models performing poorly**
- Check data quality and preprocessing
- Try different hyperparameters
- Use more cross-validation folds
- Consider feature engineering

**Issue: Training is slow**
- Reduce number of CV folds
- Limit n_trials for hyperparameter optimization
- Use fewer estimators for gradient boosting models
- Consider using a subset of data for quick prototyping

## Next Steps

- Read the [ROADMAP.md](../ROADMAP.md) for upcoming features
- Check out example notebooks in `examples/` directory
- Explore the [CLAUDE.md](../CLAUDE.md) for development commands
- Join our community discussions for support

## Contributing

Want to add a new model to the ML engine? It's easy!

1. Inherit from `BaseClassifier` or `BaseRegressor`
2. Implement `fit()` and `predict()` methods
3. Register your model: `model_registry.register("your_model", YourModel, "classification")`
4. Add tests in `tests/unit/test_ml_engine_*.py`

See existing models in `src/core/ml_engine/classical.py` for examples.
