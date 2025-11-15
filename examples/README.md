# ML Engine Examples

This directory contains examples and tutorials demonstrating the custom ML engine capabilities.

## Files

### `ml_engine_demo.py`
**Comprehensive command-line demonstration**

A standalone Python script that demonstrates the complete ML engine workflow:
- Classification and regression examples
- AutoML pipeline usage
- Model comparison and selection
- Feature importance extraction
- Individual model usage

**Run it:**
```bash
# Make sure ML dependencies are installed
uv sync --extra ml

# Run the demo
python examples/ml_engine_demo.py
```

### `ml_engine_tutorial.ipynb`
**Interactive Jupyter notebook tutorial**

Step-by-step interactive tutorial covering:
1. Model registry and discovery
2. AutoML pipeline for model comparison
3. Model evaluation with metrics and visualizations
4. Feature importance analysis
5. Hyperparameter optimization with Optuna
6. Using individual models directly
7. Sklearn compatibility demonstrations

**Run it:**
```bash
# Install Jupyter if not already installed
uv add jupyter matplotlib seaborn

# Start Jupyter
jupyter notebook examples/ml_engine_tutorial.ipynb
```

## Quick Start

### Option 1: Command Line Demo (Fastest)
```bash
python examples/ml_engine_demo.py
```

### Option 2: Interactive Tutorial (Best for Learning)
```bash
jupyter notebook examples/ml_engine_tutorial.ipynb
```

## What You'll Learn

Both examples demonstrate:

✅ **AutoML Pipeline**
- Automated model comparison
- Cross-validation strategies
- Best model selection

✅ **Model Evaluation**
- Performance metrics (accuracy, R², etc.)
- Confusion matrices
- Test set evaluation

✅ **Feature Importance**
- Understanding model decisions
- Feature ranking
- Visualization

✅ **Hyperparameter Optimization**
- Optuna integration
- Automated parameter tuning
- Performance improvement

✅ **Individual Models**
- XGBoost, LightGBM, CatBoost usage
- Custom parameter configuration
- Direct model control

✅ **Sklearn Compatibility**
- Pipeline integration
- Cross-validation
- Grid search compatibility

## Requirements

```bash
# Install all ML dependencies
uv sync --extra ml

# OR install individually
pip install xgboost lightgbm catboost optuna scikit-learn pandas numpy

# For Jupyter notebook
pip install jupyter matplotlib seaborn
```

## Example Output

The demos will show you:

```
╔══════════════════════════════════════════════════════════════╗
║           ML ENGINE END-TO-END DEMONSTRATION                 ║
╚══════════════════════════════════════════════════════════════╝

📊 Generating sample classification dataset...
   Dataset shape: (500, 15)
   Class distribution: {0: 300, 1: 200}

📈 Comparing models with 5-fold cross-validation...

🏆 Model Comparison Results:
          model  cv_mean  cv_std  test_score
  xgboost_classifier    0.92    0.03        0.94
 lightgbm_classifier    0.91    0.02        0.93
 catboost_classifier    0.90    0.04        0.92

✨ Best model selected: xgboost_classifier
   Training score: 0.9200 (±0.0300)

📊 Test set predictions (first 10):
   True values:  [1 0 1 0 1 1 0 0 1 0]
   Predictions:  [1 0 1 0 1 1 0 0 1 0]

   Test accuracy: 0.9400

🔍 Top 10 Most Important Features:
        feature  importance
      feature_0       0.250
      feature_1       0.180
      feature_2       0.150
      ...
```

## Next Steps

After running these examples:

1. 📚 Read [docs/ML_ENGINE_GUIDE.md](../docs/ML_ENGINE_GUIDE.md) for comprehensive documentation
2. 🔬 Try with your own datasets
3. ⚙️ Experiment with hyperparameter optimization
4. 🎯 Integrate with MLflow for experiment tracking
5. 🚀 Deploy your best models

## Troubleshooting

**Import errors:**
```bash
# Make sure you're in the project root and dependencies are installed
cd /path/to/magik_merlin_ml_platform
uv sync --extra ml
```

**ModuleNotFoundError: No module named 'src':**
```bash
# Make sure you're running from the project root
cd /path/to/magik_merlin_ml_platform
python examples/ml_engine_demo.py
```

**Optuna not found:**
```bash
# Install ML extras which includes Optuna
uv sync --extra ml
```

## Contributing

Have a cool example or use case? We'd love to see it! Feel free to:
- Add your own example scripts
- Enhance existing examples
- Share interesting datasets or problems
- Create tutorials for specific use cases

## Support

- 📖 [ML Engine Guide](../docs/ML_ENGINE_GUIDE.md)
- 📋 [README](../README.md)
- 🛠️ [CLAUDE.md](../CLAUDE.md) - Development commands
- 🗺️ [ROADMAP.md](../ROADMAP.md) - Future plans
