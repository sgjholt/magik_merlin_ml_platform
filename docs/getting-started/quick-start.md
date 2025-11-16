# Quick Start

Get up and running with your first ML experiment in 5 minutes.

## 1. Start the Platform

```bash
# Start the development server
./run.sh dev

# Or using Python directly
python main.py
```

The platform will be available at [http://localhost:5006](http://localhost:5006)

## 2. Start MLflow Server

MLflow is used for experiment tracking. You can start it from the UI or command line:

**From UI (Recommended):**
- Click the **🚀 Start MLflow** button in the sidebar
- Wait for the status indicator to turn green
- Click **📊 MLflow UI** to open the tracking interface

**From Command Line:**
```bash
./run.sh mlflow start
```

## 3. Load Data

### Option A: Use Demo Data

```bash
# Load built-in demo datasets
./run.sh demo
```

Then in the UI:
1. Go to the **Data Management** tab
2. Select "Local" as the data source
3. Browse to `data/demo/iris.csv`
4. Click "Load Data"

### Option B: Upload Your Data

1. Go to the **Data Management** tab
2. Click "Upload File"
3. Select a CSV file
4. Preview and confirm

## 4. Run an Experiment

### Quick AutoML

1. Go to the **Experimentation** tab
2. Click "New Experiment"
3. Configure experiment:
   - **Name**: "My First Experiment"
   - **Target Column**: Select your target variable
   - **Task Type**: Classification or Regression
4. Click "Run AutoML"

The platform will automatically:
- Split data into train/test sets
- Train multiple models (XGBoost, LightGBM, CatBoost)
- Evaluate performance with cross-validation
- Select the best model
- Log results to MLflow

### Train a Single Model

For more control:

1. Go to the **Experimentation** tab
2. Select a specific model (e.g., "XGBoost Classifier")
3. Configure hyperparameters:
   - `n_estimators`: 100
   - `max_depth`: 5
   - `learning_rate`: 0.1
4. Click "Train Model"

## 5. View Results

### In the Platform

1. Go to the **Evaluation** tab
2. View model comparison table
3. Check feature importance charts
4. Inspect predictions

### In MLflow UI

1. Click **📊 MLflow UI** in the sidebar
2. Browse experiments and runs
3. Compare metrics across runs
4. View artifacts (models, plots)

## Example: Classification Task

Here's a complete example using the Iris dataset:

```python
from src.core.ml_engine import AutoMLPipeline
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create AutoML pipeline
pipeline = AutoMLPipeline(task_type="classification")

# Compare models
results = pipeline.compare_models(X_train, y_train, cv=5)
print(results)

# Get best model
best_model = pipeline.get_best_model()

# Make predictions
predictions = best_model.predict(X_test)

# Evaluate
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy: {accuracy:.3f}")
```

## Next Steps

### Learn More

- [Configuration Guide](configuration.md) - Set up data sources and MLflow
- [ML Engine Guide](../user-guide/ml-engine.md) - Use the custom ML engine
- [AutoML Guide](../user-guide/automl.md) - Automated model selection
- [Experiments Guide](../user-guide/experiments.md) - Track and compare experiments

### Try Advanced Features

- **Hyperparameter Optimization**: Use Optuna for automated tuning
- **Custom Models**: Add your own model wrappers
- **Data Sources**: Connect to Snowflake or AWS S3
- **Visualizations**: Create custom plots and charts

### Common Commands

```bash
# Platform management
./run.sh dev          # Start development server
./run.sh status       # Check platform status
./run.sh health       # Run health checks

# MLflow management
./run.sh mlflow start # Start MLflow server
./run.sh mlflow stop  # Stop MLflow server
./run.sh mlflow ui    # Open MLflow web UI

# Development
./run.sh test         # Run tests
./run.sh lint         # Check code quality
./run.sh format       # Format code
./run.sh coverage     # Generate coverage report
```

## Troubleshooting

### Platform Won't Start

**Check if port 5006 is in use:**
```bash
lsof -i :5006  # macOS/Linux
netstat -ano | findstr :5006  # Windows
```

**Use a different port:**
```bash
./run.sh dev PORT=8080
```

### MLflow Connection Failed

**Check MLflow status:**
```bash
./run.sh mlflow status
```

**Restart MLflow:**
```bash
./run.sh mlflow restart
```

### Data Loading Error

**Verify file format:**
- Must be CSV, Parquet, or Excel
- First row should contain column headers
- No empty columns

**Check file permissions:**
```bash
ls -l data/your-file.csv
chmod 644 data/your-file.csv
```

## Getting Help

- **Documentation**: Browse the [User Guide](../user-guide/overview.md)
- **Examples**: Check `examples/` directory for sample scripts
- **Issues**: Report problems on [GitHub](https://github.com/sgjholt/magik_merlin_ml_platform/issues)
- **Discussions**: Ask questions in [GitHub Discussions](https://github.com/sgjholt/magik_merlin_ml_platform/discussions)
