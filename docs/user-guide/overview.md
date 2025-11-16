# User Guide Overview

Welcome to the Magik Merlin ML Platform user guide. This section provides comprehensive documentation for all platform features.

## Contents

- **[Data Management](data-management.md)** - Load and manage datasets
- **[ML Engine](ml-engine.md)** - Use the custom ML engine
- **[AutoML](automl.md)** - Automated model selection and optimization
- **[Experiments](experiments.md)** - Track and compare experiments
- **[Visualization](visualization.md)** - Create interactive charts
- **[Deployment](deployment.md)** - Deploy models to production

## Platform Overview

The ML Platform provides an end-to-end workflow for machine learning:

1. **Data Ingestion** → Load from multiple sources
2. **Experimentation** → Train and compare models
3. **Evaluation** → Analyze results and feature importance
4. **Tracking** → Log experiments with MLflow
5. **Deployment** → Deploy best models to production

## Key Concepts

### Experiments

An experiment is a collection of model training runs with the same configuration. Each run tracks:
- Model hyperparameters
- Training metrics (accuracy, loss, etc.)
- Validation metrics
- Feature importance
- Model artifacts

### Models

The platform supports multiple model types:
- **XGBoost**: Gradient boosting (classification/regression)
- **LightGBM**: Fast gradient boosting
- **CatBoost**: Handles categorical features natively
- **PyTorch**: Deep learning (coming in Phase 3)

### AutoML

Automated machine learning that:
- Compares multiple models automatically
- Tunes hyperparameters with Optuna
- Selects the best model based on CV scores
- Logs all runs to MLflow

## Common Workflows

### Workflow 1: Quick AutoML

For rapid prototyping:

1. Load data
2. Run AutoML
3. Review results
4. Deploy best model

### Workflow 2: Custom Model Training

For fine-grained control:

1. Load data
2. Select specific model
3. Configure hyperparameters
4. Train and evaluate
5. Iterate and improve

### Workflow 3: Hyperparameter Optimization

For best performance:

1. Load data
2. Select model
3. Define parameter search space
4. Run Optuna optimization
5. Train with best parameters

## Best Practices

- **Use AutoML first** to get a baseline
- **Track all experiments** with meaningful names
- **Use cross-validation** for reliable metrics
- **Monitor feature importance** to understand models
- **Compare multiple runs** before deployment

## Next Steps

Start with [Data Management](data-management.md) to load your first dataset.
