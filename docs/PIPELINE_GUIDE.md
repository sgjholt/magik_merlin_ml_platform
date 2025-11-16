# Pipeline System Guide

## Overview

The ML Platform's Pipeline System provides a comprehensive framework for building, executing, scheduling, and managing ML workflows. Pipelines are composed of interconnected nodes that perform specific tasks in a defined order.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [Creating Pipelines](#creating-pipelines)
3. [Available Nodes](#available-nodes)
4. [Executing Pipelines](#executing-pipelines)
5. [Scheduling Pipelines](#scheduling-pipelines)
6. [Storage and Versioning](#storage-and-versioning)
7. [UI Guide](#ui-guide)
8. [Advanced Topics](#advanced-topics)
9. [Examples](#examples)

---

## Core Concepts

### Pipelines

A **Pipeline** is a directed acyclic graph (DAG) of nodes that process data sequentially or in parallel. Each pipeline has:

- **Unique ID**: Identifier for the pipeline
- **Name**: Human-readable name
- **Description**: Description of what the pipeline does
- **Nodes**: Processing units that perform specific tasks
- **Edges**: Connections between nodes that define data flow
- **Status**: Current state (pending, running, completed, failed, cancelled)

### Nodes

**Nodes** are the building blocks of pipelines. Each node:

- Performs a specific task (data loading, preprocessing, training, etc.)
- Has inputs and outputs
- Can be connected to other nodes via edges
- Tracks execution metrics (time, memory, rows processed)
- Has a status (pending, running, completed, failed, skipped)

### Execution

The **PipelineExecutor** runs pipelines by:

1. Validating the pipeline structure
2. Computing the execution order (topological sort)
3. Executing nodes in order
4. Passing data between nodes
5. Tracking progress and handling errors

---

## Creating Pipelines

### Programmatic API

```python
from src.core.pipeline_orchestration import (
    Pipeline,
    DataLoaderNode,
    TrainTestSplitNode,
    ModelTrainerNode,
)

# Create a new pipeline
pipeline = Pipeline(
    pipeline_id="ml_pipeline_001",
    name="Customer Churn Prediction",
    description="Train a model to predict customer churn"
)

# Add nodes
loader = DataLoaderNode(
    node_id="load_data",
    source_type="file",
    source_path="data/customers.csv"
)

splitter = TrainTestSplitNode(
    node_id="split_data",
    test_size=0.2,
    target_column="churn",
    random_state=42
)

trainer = ModelTrainerNode(
    node_id="train_model",
    model_type="xgboost",
    task_type="classification",
    model_params={"n_estimators": 100, "max_depth": 5}
)

# Add nodes to pipeline
pipeline.add_node(loader)
pipeline.add_node(splitter)
pipeline.add_node(trainer)

# Connect nodes
pipeline.add_edge("load_data", "split_data")
pipeline.add_edge("split_data", "train_model")

# Validate pipeline
is_valid, errors = pipeline.validate()
if not is_valid:
    print(f"Pipeline validation failed: {errors}")
```

### UI-Based Creation

1. Navigate to the **Pipelines** tab
2. Click **➕ New Pipeline**
3. Enter pipeline name and description
4. Use the **Add Node** button to add nodes
5. Nodes are automatically connected in sequence
6. Click **▶️ Execute Pipeline** to run

---

## Available Nodes

### 1. DataLoaderNode

Loads data from various sources.

**Configuration:**
- `source_type`: Type of source ("file", "database", etc.)
- `source_path`: Path to data file

**Supported Formats:**
- CSV (`.csv`)
- Parquet (`.parquet`)
- Excel (`.xlsx`, `.xls`)

**Example:**
```python
loader = DataLoaderNode(
    node_id="loader",
    source_type="file",
    source_path="data/train.csv"
)
```

### 2. DataPreprocessorNode

Performs data preprocessing operations.

**Configuration:**
- `operations`: List of preprocessing steps

**Available Operations:**
- `drop_missing`: Remove rows with missing values
- `fill_missing_mean`: Fill missing values with column mean
- `remove_duplicates`: Remove duplicate rows

**Example:**
```python
preprocessor = DataPreprocessorNode(
    node_id="preprocess",
    operations=["drop_missing", "remove_duplicates"]
)
```

### 3. TrainTestSplitNode

Splits data into training and test sets.

**Configuration:**
- `test_size`: Proportion of data for test set (0.0-1.0)
- `target_column`: Name of the target column
- `random_state`: Random seed for reproducibility

**Example:**
```python
splitter = TrainTestSplitNode(
    node_id="split",
    test_size=0.2,
    target_column="target",
    random_state=42
)
```

### 4. FeatureScalerNode

Scales features using StandardScaler.

**Configuration:**
- No required configuration (uses sklearn's StandardScaler)

**Example:**
```python
scaler = FeatureScalerNode(node_id="scale")
```

### 5. ModelTrainerNode

Trains ML models.

**Configuration:**
- `model_type`: Type of model ("xgboost", "lightgbm", "catboost", "lightning")
- `task_type`: "classification" or "regression"
- `model_params`: Dictionary of model hyperparameters

**Example:**
```python
trainer = ModelTrainerNode(
    node_id="trainer",
    model_type="xgboost",
    task_type="classification",
    model_params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
    }
)
```

### 6. ModelEvaluatorNode

Evaluates trained models on test data.

**Configuration:**
- No required configuration

**Example:**
```python
evaluator = ModelEvaluatorNode(node_id="evaluate")
```

### 7. ModelSaverNode

Saves trained models to disk.

**Configuration:**
- `save_path`: Path where model should be saved

**Example:**
```python
saver = ModelSaverNode(
    node_id="save",
    save_path="models/churn_model.pkl"
)
```

---

## Executing Pipelines

### Synchronous Execution

```python
from src.core.pipeline_orchestration import PipelineExecutor

executor = PipelineExecutor()

# Execute pipeline synchronously (blocking)
result = executor.execute(pipeline)

print(f"Status: {result.status}")
print(f"Duration: {result.duration}s")
print(f"Errors: {result.error}")
```

### Asynchronous Execution

```python
# Execute in background thread
def on_progress(pipeline_id, node_id, progress_pct):
    print(f"Progress: {progress_pct:.1f}% - Executing {node_id}")

result = executor.execute(
    pipeline,
    async_mode=True,
    progress_callback=on_progress
)

# Check if still running
is_running = executor.is_running(pipeline.pipeline_id)

# Get result when complete
final_result = executor.get_result(pipeline.pipeline_id)
```

### Cancelling Execution

```python
# Cancel a running pipeline
executor.cancel_pipeline(pipeline.pipeline_id)
```

---

## Scheduling Pipelines

### Cron-Based Scheduling

```python
from src.core.pipeline_orchestration import PipelineScheduler, ScheduleConfig

scheduler = PipelineScheduler(executor)

# Schedule for daily execution at midnight
schedule = ScheduleConfig(
    schedule_type="cron",
    cron_expression="0 0 * * *",  # Daily at midnight
    enabled=True
)

scheduler.schedule_pipeline(pipeline, schedule)

# Start the scheduler
scheduler.start()
```

### Interval-Based Scheduling

```python
# Run every hour
schedule = ScheduleConfig(
    schedule_type="interval",
    interval_seconds=3600,  # 1 hour
    enabled=True
)

scheduler.schedule_pipeline(pipeline, schedule)
```

### One-Time Execution

```python
from datetime import datetime, timedelta

# Schedule for one-time execution in 1 hour
schedule = ScheduleConfig(
    schedule_type="once",
    start_time=datetime.now() + timedelta(hours=1),
    enabled=True
)

scheduler.schedule_pipeline(pipeline, schedule)
```

### Managing Schedules

```python
# List all scheduled pipelines
scheduled_pipelines = scheduler.list_scheduled_pipelines()

# Get specific schedule
schedule_info = scheduler.get_schedule(pipeline.pipeline_id)

# Unschedule pipeline
scheduler.unschedule_pipeline(pipeline.pipeline_id)

# Stop scheduler
scheduler.stop()
```

---

## Storage and Versioning

### Saving Pipelines

```python
from src.core.pipeline_orchestration import PipelineStorage

storage = PipelineStorage(storage_dir="data/pipelines")

# Save pipeline
storage.save_pipeline(pipeline)

# Save with versioning
storage.save_pipeline(pipeline, create_version=True)
```

### Loading Pipelines

```python
# Load pipeline by ID
loaded_pipeline = storage.load_pipeline("ml_pipeline_001")

# List all pipelines
pipelines = storage.list_pipelines()

for p in pipelines:
    print(f"{p['name']}: {p['status']} ({p['nodes']} nodes)")
```

### Version Management

```python
# List versions of a pipeline
versions = storage.list_versions(pipeline.pipeline_id)

for version in versions:
    print(f"Version {version.version}: {version.created_at}")

# Restore to specific version
restored = storage.restore_version(pipeline.pipeline_id, "v2")
```

### Execution History

```python
# Save execution result
storage.save_execution_result(result)

# List executions for a pipeline
executions = storage.list_executions(pipeline.pipeline_id, limit=10)

for exec_data in executions:
    print(f"{exec_data['start_time']}: {exec_data['status']}")
```

---

## UI Guide

### Pipeline Management Panel

The **🔄 Pipelines** tab provides a visual interface for managing pipelines.

#### Creating a Pipeline

1. Click **➕ New Pipeline**
2. Enter **Pipeline Name** and **Description**
3. Select **Node Type** from dropdown
4. Click **➕ Add Node** to add nodes
5. Nodes are automatically connected in sequence
6. Click **▶️ Execute Pipeline** when ready

#### Executing a Pipeline

1. Select a pipeline from the list
2. Click **▶️ Execute Pipeline**
3. Monitor progress in the progress bar
4. View execution status in the status text
5. Check **Execution History** for past runs

#### Scheduling a Pipeline

1. Select a pipeline from the list
2. Click **📅 Schedule**
3. Pipeline is scheduled for daily execution
4. View scheduled pipelines in the scheduler

#### Viewing History

The **Execution History** table shows:
- **Time**: When the pipeline was executed
- **Status**: Completed, failed, or cancelled
- **Duration**: How long it took to run

---

## Advanced Topics

### Custom Nodes

Create custom nodes by extending `BaseNode`:

```python
from src.core.pipeline_orchestration.nodes import BaseNode, NodeOutput, NodeType

class CustomTransformNode(BaseNode):
    def __init__(self, node_id: str, transform_fn=None, **kwargs):
        super().__init__(
            node_id=node_id,
            node_type=NodeType.CUSTOM,
            **kwargs
        )
        self.transform_fn = transform_fn

    def execute(self, inputs):
        data = inputs["data"].data
        transformed = self.transform_fn(data)
        return NodeOutput(data=transformed)
```

### Complex Workflows

Build complex pipelines with multiple branches:

```python
# Create pipeline with parallel branches
pipeline = Pipeline(pipeline_id="complex", name="Complex Pipeline")

# Branch 1: XGBoost
trainer_xgb = ModelTrainerNode(
    node_id="train_xgb",
    model_type="xgboost"
)

# Branch 2: LightGBM
trainer_lgb = ModelTrainerNode(
    node_id="train_lgb",
    model_type="lightgbm"
)

# Both branches start from same preprocessor
pipeline.add_edge("preprocess", "train_xgb")
pipeline.add_edge("preprocess", "train_lgb")
```

### Error Handling

Handle pipeline errors gracefully:

```python
result = executor.execute(pipeline)

if result.status == PipelineStatus.FAILED:
    print(f"Pipeline failed: {result.error}")

    # Check which nodes failed
    for node_id, error in result.node_errors.items():
        print(f"Node {node_id} failed: {error}")
```

---

## Examples

### Example 1: Simple Data Processing Pipeline

```python
from src.core.pipeline_orchestration import *

# Create pipeline
pipeline = Pipeline(
    pipeline_id="data_pipeline",
    name="Data Processing Pipeline"
)

# Add nodes
loader = DataLoaderNode(
    node_id="load",
    source_type="file",
    source_path="data/raw.csv"
)

preprocessor = DataPreprocessorNode(
    node_id="clean",
    operations=["drop_missing", "remove_duplicates"]
)

pipeline.add_node(loader)
pipeline.add_node(preprocessor)
pipeline.add_edge("load", "clean")

# Execute
executor = PipelineExecutor()
result = executor.execute(pipeline)
```

### Example 2: Complete ML Training Pipeline

```python
# Create comprehensive ML pipeline
pipeline = Pipeline(
    pipeline_id="ml_training",
    name="ML Training Pipeline"
)

# Data loading
loader = DataLoaderNode(
    node_id="load",
    source_type="file",
    source_path="data/customers.csv"
)

# Preprocessing
preprocessor = DataPreprocessorNode(
    node_id="preprocess",
    operations=["fill_missing_mean", "remove_duplicates"]
)

# Split
splitter = TrainTestSplitNode(
    node_id="split",
    test_size=0.2,
    target_column="churn",
    random_state=42
)

# Scale
scaler = FeatureScalerNode(node_id="scale")

# Train
trainer = ModelTrainerNode(
    node_id="train",
    model_type="xgboost",
    task_type="classification",
    model_params={
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1
    }
)

# Evaluate
evaluator = ModelEvaluatorNode(node_id="evaluate")

# Save
saver = ModelSaverNode(
    node_id="save",
    save_path="models/churn_model.pkl"
)

# Add nodes
for node in [loader, preprocessor, splitter, scaler, trainer, evaluator, saver]:
    pipeline.add_node(node)

# Connect nodes
pipeline.add_edge("load", "preprocess")
pipeline.add_edge("preprocess", "split")
pipeline.add_edge("split", "scale")
pipeline.add_edge("scale", "train")
pipeline.add_edge("train", "evaluate")
pipeline.add_edge("evaluate", "save")

# Execute and save
executor = PipelineExecutor()
result = executor.execute(pipeline)

if result.status == PipelineStatus.COMPLETED:
    storage = PipelineStorage()
    storage.save_pipeline(pipeline)
    storage.save_execution_result(result)
    print("Pipeline completed successfully!")
```

### Example 3: Scheduled Pipeline

```python
# Create pipeline
pipeline = Pipeline(
    pipeline_id="daily_retrain",
    name="Daily Model Retraining"
)

# ... add nodes ...

# Schedule for daily execution
scheduler = PipelineScheduler(executor)
schedule = ScheduleConfig(
    schedule_type="cron",
    cron_expression="0 2 * * *",  # Daily at 2 AM
    enabled=True,
    max_retries=3,
    retry_delay_seconds=300  # 5 minutes
)

scheduler.schedule_pipeline(pipeline, schedule)
scheduler.start()

# Scheduler will run pipeline automatically
```

---

## Best Practices

1. **Use Descriptive Names**: Give pipelines and nodes meaningful names
2. **Validate Early**: Always call `pipeline.validate()` before execution
3. **Handle Errors**: Check execution results and handle failures gracefully
4. **Version Pipelines**: Use `create_version=True` when making changes
5. **Monitor Progress**: Use progress callbacks for long-running pipelines
6. **Save Results**: Always save execution results for tracking
7. **Test Incrementally**: Test nodes individually before building complex pipelines
8. **Use Async Mode**: Use `async_mode=True` for long-running pipelines in production

---

## Troubleshooting

### Pipeline Validation Fails

- Check that all nodes are connected
- Ensure no cycles exist in the graph
- Verify all required node parameters are provided

### Pipeline Execution Hangs

- Use async mode with progress callback to monitor progress
- Check logs for specific node errors
- Ensure data files exist and are accessible

### Out of Memory Errors

- Process data in chunks
- Use data sampling for testing
- Increase system memory or use distributed processing

---

## API Reference

For detailed API documentation, see:
- `src/core/pipeline_orchestration/nodes.py` - Node implementations
- `src/core/pipeline_orchestration/pipeline.py` - Pipeline class
- `src/core/pipeline_orchestration/executor.py` - Execution engine
- `src/core/pipeline_orchestration/scheduler.py` - Scheduling system
- `src/core/pipeline_orchestration/storage.py` - Storage and versioning

---

## Support

For issues or questions:
- Check the logs in `logs/` directory
- Review test files in `tests/unit/test_pipeline_*.py`
- Consult the main documentation in `README.md`
