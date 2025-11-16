# AutoML

Automated machine learning for rapid model development.

## Overview

The AutoML pipeline automatically:
- Trains multiple models (XGBoost, LightGBM, CatBoost)
- Performs cross-validation
- Compares performance metrics
- Selects the best model
- Logs everything to MLflow

## Quick Start

```python
from src.core.ml_engine import AutoMLPipeline

# Create pipeline
pipeline = AutoMLPipeline(task_type="classification")

# Compare models
results = pipeline.compare_models(X_train, y_train, cv=5)

# Get best model
best_model = pipeline.get_best_model()
```

See [ML Engine Guide](ml-engine.md) for complete documentation.
