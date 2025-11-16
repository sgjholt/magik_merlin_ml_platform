"""
Pipeline System Demo

This script demonstrates the core functionality of the ML Platform's
pipeline system, including:
- Creating pipelines
- Adding and connecting nodes
- Executing pipelines
- Scheduling pipelines
- Saving and loading pipelines
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.pipeline_orchestration import (
    DataLoaderNode,
    DataPreprocessorNode,
    FeatureScalerNode,
    ModelEvaluatorNode,
    ModelTrainerNode,
    Pipeline,
    PipelineExecutor,
    PipelineScheduler,
    PipelineStorage,
    ScheduleConfig,
    TrainTestSplitNode,
)


def create_sample_data(output_dir: Path) -> Path:
    """Create sample dataset for demo."""
    print("Creating sample dataset...")

    # Generate synthetic customer churn data
    np.random.seed(42)
    n_samples = 200

    data = pd.DataFrame(
        {
            "tenure": np.random.randint(1, 72, n_samples),
            "monthly_charges": np.random.uniform(20, 120, n_samples),
            "total_charges": np.random.uniform(100, 8000, n_samples),
            "contract_type": np.random.choice([0, 1, 2], n_samples),
            "payment_method": np.random.choice([0, 1, 2, 3], n_samples),
            "churn": np.random.randint(0, 2, n_samples),
        }
    )

    # Save to CSV
    data_path = output_dir / "customer_data.csv"
    data.to_csv(data_path, index=False)

    print(f"✓ Created dataset with {len(data)} samples")
    print(f"✓ Saved to: {data_path}")

    return data_path


def demo_simple_pipeline(data_path: Path) -> None:
    """Demonstrate a simple data processing pipeline."""
    print("\n" + "=" * 60)
    print("Demo 1: Simple Data Processing Pipeline")
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline(
        pipeline_id="simple_pipeline",
        name="Simple Data Processing",
        description="Load and preprocess data",
    )

    # Add nodes
    loader = DataLoaderNode(
        node_id="load_data",
        source_type="file",
        source_path=str(data_path),
        name="Load Customer Data",
    )

    preprocessor = DataPreprocessorNode(
        node_id="preprocess",
        operations=["drop_missing", "remove_duplicates"],
        name="Clean Data",
    )

    pipeline.add_node(loader)
    pipeline.add_node(preprocessor)
    pipeline.add_edge("load_data", "preprocess")

    # Validate
    is_valid, errors = pipeline.validate()
    if not is_valid:
        print(f"❌ Pipeline validation failed: {errors}")
        return

    print("✓ Pipeline validated successfully")

    # Execute
    print("\nExecuting pipeline...")
    executor = PipelineExecutor()
    result = executor.execute(pipeline)

    print(f"\n{'Status:':<20} {result.status.value}")
    print(f"{'Duration:':<20} {result.duration:.2f}s")
    print(f"{'Nodes executed:':<20} {len(result.node_outputs)}")

    if result.status.value == "completed":
        print("✓ Pipeline completed successfully!")


def demo_ml_pipeline(data_path: Path, output_dir: Path) -> None:
    """Demonstrate a complete ML training pipeline."""
    print("\n" + "=" * 60)
    print("Demo 2: Complete ML Training Pipeline")
    print("=" * 60)

    # Create pipeline
    pipeline = Pipeline(
        pipeline_id="ml_training_pipeline",
        name="Customer Churn Prediction Model",
        description="Train and evaluate a churn prediction model",
    )

    # Build pipeline nodes
    loader = DataLoaderNode(
        node_id="load_data",
        source_type="file",
        source_path=str(data_path),
        name="Load Training Data",
    )

    preprocessor = DataPreprocessorNode(
        node_id="preprocess",
        operations=["fill_missing_mean", "remove_duplicates"],
        name="Preprocess Data",
    )

    splitter = TrainTestSplitNode(
        node_id="split",
        test_size=0.2,
        target_column="churn",
        random_state=42,
        name="Train-Test Split",
    )

    scaler = FeatureScalerNode(
        node_id="scale",
        name="Scale Features",
    )

    trainer = ModelTrainerNode(
        node_id="train",
        model_type="xgboost",
        task_type="classification",
        model_params={
            "n_estimators": 50,
            "max_depth": 4,
            "learning_rate": 0.1,
        },
        name="Train XGBoost Model",
    )

    evaluator = ModelEvaluatorNode(
        node_id="evaluate",
        name="Evaluate Model",
    )

    # Add nodes to pipeline
    print("\nBuilding pipeline...")
    for node in [loader, preprocessor, splitter, scaler, trainer, evaluator]:
        pipeline.add_node(node)
        print(f"  + Added node: {node.name}")

    # Connect nodes
    print("\nConnecting nodes...")
    edges = [
        ("load_data", "preprocess"),
        ("preprocess", "split"),
        ("split", "scale"),
        ("scale", "train"),
        ("train", "evaluate"),
    ]

    for source, target in edges:
        pipeline.add_edge(source, target)
        print(f"  → {source} → {target}")

    # Validate
    is_valid, errors = pipeline.validate()
    if not is_valid:
        print(f"\n❌ Pipeline validation failed: {errors}")
        return

    print("\n✓ Pipeline validated successfully")

    # Execute with progress tracking
    print("\nExecuting pipeline...")

    def on_progress(pipeline_id: str, node_id: str, progress: float) -> None:
        print(f"  Progress: {progress:.1f}% - Executing {node_id}")

    executor = PipelineExecutor()
    result = executor.execute(pipeline, progress_callback=on_progress)

    # Display results
    print(f"\n{'=' * 60}")
    print("Execution Results")
    print("=" * 60)
    print(f"{'Status:':<20} {result.status.value}")
    print(f"{'Duration:':<20} {result.duration:.2f}s")
    print(f"{'Nodes executed:':<20} {len(result.node_outputs)}")

    if result.status.value == "completed":
        # Get evaluation results
        eval_output = result.node_outputs["evaluate"]
        test_score = eval_output.data["test_score"]

        print(f"{'Test Score:':<20} {test_score:.4f}")
        print("\n✓ ML pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {result.error}")


def demo_pipeline_storage(pipeline: Pipeline, output_dir: Path) -> None:
    """Demonstrate pipeline storage and versioning."""
    print("\n" + "=" * 60)
    print("Demo 3: Pipeline Storage and Versioning")
    print("=" * 60)

    # Initialize storage
    storage_dir = output_dir / "pipeline_storage"
    storage = PipelineStorage(storage_dir=storage_dir)

    print(f"\nStorage directory: {storage_dir}")

    # Save pipeline
    print("\nSaving pipeline...")
    storage.save_pipeline(pipeline)
    print("✓ Pipeline saved")

    # List pipelines
    pipelines = storage.list_pipelines()
    print(f"\nStored pipelines: {len(pipelines)}")
    for p in pipelines:
        print(f"  - {p['name']} ({p['nodes']} nodes)")

    # Load pipeline
    print("\nLoading pipeline...")
    loaded = storage.load_pipeline(pipeline.pipeline_id)
    print(f"✓ Loaded: {loaded.name}")
    print(f"  Nodes: {len(loaded.nodes)}")
    print(f"  Edges: {len(loaded.edges)}")

    # Get storage stats
    stats = storage.get_storage_stats()
    print("\nStorage Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def demo_pipeline_scheduling() -> None:
    """Demonstrate pipeline scheduling."""
    print("\n" + "=" * 60)
    print("Demo 4: Pipeline Scheduling")
    print("=" * 60)

    # Create a simple pipeline
    pipeline = Pipeline(
        pipeline_id="scheduled_pipeline",
        name="Daily Data Processing",
    )

    # Add a simple node
    from datetime import datetime, timedelta

    loader = DataLoaderNode(
        node_id="loader",
        source_type="file",
        source_path="/path/to/daily_data.csv",
    )
    pipeline.add_node(loader)

    # Initialize scheduler
    executor = PipelineExecutor()
    scheduler = PipelineScheduler(executor)

    # Create schedule configurations
    schedules = [
        (
            "Daily at midnight",
            ScheduleConfig(
                schedule_type="cron",
                cron_expression="0 0 * * *",
                enabled=False,  # Disabled for demo
            ),
        ),
        (
            "Every hour",
            ScheduleConfig(
                schedule_type="interval",
                interval_seconds=3600,
                enabled=False,  # Disabled for demo
            ),
        ),
        (
            "One-time in 1 hour",
            ScheduleConfig(
                schedule_type="once",
                start_time=datetime.now() + timedelta(hours=1),
                enabled=False,  # Disabled for demo
            ),
        ),
    ]

    print("\nAvailable schedule types:")
    for desc, schedule in schedules:
        print(f"  - {desc}")
        print(f"    Type: {schedule.schedule_type}")
        if schedule.cron_expression:
            print(f"    Cron: {schedule.cron_expression}")
        if schedule.interval_seconds:
            print(f"    Interval: {schedule.interval_seconds}s")

    # Note: Not actually scheduling in demo to avoid background processes
    print("\n(Scheduling disabled in demo mode)")


def main() -> None:
    """Run all pipeline demos."""
    print("=" * 60)
    print("ML Platform Pipeline System Demo")
    print("=" * 60)

    # Create temporary directory for demo
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Create sample data
        data_path = create_sample_data(tmp_path)

        # Run demos
        demo_simple_pipeline(data_path)
        demo_ml_pipeline(data_path, tmp_path)

        # Create a pipeline for storage demo
        pipeline = Pipeline(
            pipeline_id="demo_pipeline",
            name="Demo Pipeline",
        )
        pipeline.add_node(
            DataLoaderNode(
                node_id="loader",
                source_type="file",
                source_path=str(data_path),
            )
        )

        demo_pipeline_storage(pipeline, tmp_path)
        demo_pipeline_scheduling()

        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        print("\nFor more information, see:")
        print("  - docs/PIPELINE_GUIDE.md - Comprehensive pipeline documentation")
        print("  - tests/unit/test_pipeline_*.py - Pipeline test examples")


if __name__ == "__main__":
    main()
