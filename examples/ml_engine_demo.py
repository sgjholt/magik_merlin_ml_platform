#!/usr/bin/env python3
"""
ML Engine End-to-End Demo

This script demonstrates the complete workflow of the custom ML engine:
1. Generate sample data
2. Compare multiple models
3. Optimize best model hyperparameters
4. Make predictions
5. Extract feature importance

Requirements:
    pip install xgboost lightgbm catboost optuna scikit-learn pandas numpy
    # OR
    uv sync --extra ml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from core.ml_engine import AutoMLPipeline


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_classification():
    """Demonstrate classification workflow."""
    print_section("CLASSIFICATION DEMO")

    # 1. Generate sample data
    print("📊 Generating sample classification dataset...")
    X, y = make_classification(
        n_samples=500,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        random_state=42,
        flip_y=0.05,  # Add some noise
    )

    # Convert to DataFrame with meaningful names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Class distribution: {y_series.value_counts().to_dict()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42, stratify=y
    )

    # 2. Create AutoML pipeline
    print_section("STEP 1: Model Comparison")
    print("🤖 Creating AutoML pipeline...")
    pipeline = AutoMLPipeline(task_type="classification", random_state=42)

    # 3. Compare models
    print("📈 Comparing models with 5-fold cross-validation...")
    results = pipeline.compare_models(X_train, y_train, cv=5, test_size=0.2)

    print("\n🏆 Model Comparison Results:")
    print(results.to_string(index=False))

    # 4. Get best model
    print_section("STEP 2: Best Model Selection")
    best_model = pipeline.get_best_model()
    print(f"✨ Best model selected: {pipeline.best_model_name}")
    print(f"   Training score: {results.iloc[0]['cv_mean']:.4f} (±{results.iloc[0]['cv_std']:.4f})")

    # 5. Make predictions
    print_section("STEP 3: Making Predictions")
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)

    print(f"📊 Test set predictions (first 10):")
    print(f"   True values:  {y_test.values[:10]}")
    print(f"   Predictions:  {predictions[:10]}")
    print(f"\n   Class 0 probabilities (first 5): {probabilities[:5, 0]}")
    print(f"   Class 1 probabilities (first 5): {probabilities[:5, 1]}")

    # Test score
    test_accuracy = best_model.score(X_test, y_test)
    print(f"\n   Test accuracy: {test_accuracy:.4f}")

    # 6. Feature importance
    print_section("STEP 4: Feature Importance")
    if hasattr(best_model, "get_feature_importance"):
        importance_df = best_model.get_feature_importance()
        print("🔍 Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    # 7. Hyperparameter optimization (optional - commented out for speed)
    print_section("STEP 5: Hyperparameter Optimization (Optional)")
    print("⚙️  To optimize hyperparameters, uncomment the code below:")
    print("""
    optimization_result = pipeline.optimize_hyperparameters(
        X_train, y_train,
        model_name=pipeline.best_model_name,
        n_trials=50,
        cv=5
    )
    print(f"Best parameters: {optimization_result['best_params']}")
    print(f"Best score: {optimization_result['best_score']:.4f}")
    """)

    print("\n✅ Classification demo completed successfully!")
    return pipeline, results


def demo_regression():
    """Demonstrate regression workflow."""
    print_section("REGRESSION DEMO")

    # 1. Generate sample data
    print("📊 Generating sample regression dataset...")
    X, y = make_regression(
        n_samples=500,
        n_features=15,
        n_informative=10,
        noise=10.0,
        random_state=42,
    )

    # Convert to DataFrame
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="target")

    print(f"   Dataset shape: {X_df.shape}")
    print(f"   Target range: [{y_series.min():.2f}, {y_series.max():.2f}]")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y_series, test_size=0.2, random_state=42
    )

    # 2. Create AutoML pipeline
    print_section("STEP 1: Model Comparison")
    print("🤖 Creating AutoML pipeline for regression...")
    pipeline = AutoMLPipeline(task_type="regression", random_state=42)

    # 3. Compare models
    print("📈 Comparing regression models...")
    results = pipeline.compare_models(X_train, y_train, cv=5, test_size=0.2)

    print("\n🏆 Model Comparison Results (R² scores):")
    print(results.to_string(index=False))

    # 4. Get best model
    print_section("STEP 2: Best Model & Predictions")
    best_model = pipeline.get_best_model()
    print(f"✨ Best model selected: {pipeline.best_model_name}")
    print(f"   Training R² score: {results.iloc[0]['cv_mean']:.4f}")

    # 5. Make predictions
    predictions = best_model.predict(X_test)

    print(f"\n📊 Test set predictions (first 10):")
    comparison = pd.DataFrame({
        'True': y_test.values[:10],
        'Predicted': predictions[:10],
        'Error': (y_test.values[:10] - predictions[:10])
    })
    print(comparison.to_string(index=False))

    # Test score
    test_r2 = best_model.score(X_test, y_test)
    print(f"\n   Test R² score: {test_r2:.4f}")

    # 6. Feature importance
    print_section("STEP 3: Feature Importance")
    if hasattr(best_model, "get_feature_importance"):
        importance_df = best_model.get_feature_importance()
        print("🔍 Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))

    print("\n✅ Regression demo completed successfully!")
    return pipeline, results


def demo_individual_model():
    """Demonstrate using individual models directly."""
    print_section("INDIVIDUAL MODEL DEMO")

    try:
        from core.ml_engine import XGBoostClassifier
    except ImportError:
        print("⚠️  XGBoost not available. Skipping individual model demo.")
        return

    # Generate simple data
    print("📊 Generating simple dataset...")
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Create and train specific model
    print("\n🤖 Training XGBoost classifier with custom parameters...")
    model = XGBoostClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    # Predictions and evaluation
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print(f"   Model trained successfully!")
    print(f"   Test accuracy: {accuracy:.4f}")

    # Get parameters
    params = model.get_params()
    print(f"\n   Model parameters: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}")

    # Feature importance
    importance_df = model.get_feature_importance()
    print(f"\n🔍 Feature importance (top 5):")
    print(importance_df.head(5).to_string(index=False))

    print("\n✅ Individual model demo completed!")


def main():
    """Run all demos."""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║           ML ENGINE END-TO-END DEMONSTRATION                     ║
    ║                                                                  ║
    ║  This script demonstrates the custom ML engine capabilities:     ║
    ║  • AutoML pipeline with model comparison                         ║
    ║  • Hyperparameter optimization                                   ║
    ║  • Prediction and evaluation                                     ║
    ║  • Feature importance extraction                                 ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Run classification demo
        clf_pipeline, clf_results = demo_classification()

        # Run regression demo
        reg_pipeline, reg_results = demo_regression()

        # Run individual model demo
        demo_individual_model()

        # Final summary
        print_section("DEMO SUMMARY")
        print("✅ All demos completed successfully!")
        print("\n📚 Next Steps:")
        print("   1. Check docs/ML_ENGINE_GUIDE.md for comprehensive documentation")
        print("   2. Explore examples/ml_engine_tutorial.ipynb for interactive examples")
        print("   3. Try with your own data!")
        print("\n🚀 Happy ML Engineering!")

    except ImportError as e:
        print(f"\n❌ Error: Missing required libraries.")
        print(f"   {e}")
        print("\n💡 Install ML dependencies with:")
        print("   uv sync --extra ml")
        print("   # OR")
        print("   pip install xgboost lightgbm catboost optuna scikit-learn pandas numpy")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
