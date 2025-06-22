"""
Unit tests for PyCaret integration
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.ml.pycaret_integration import (
    PYCARET_AVAILABLE,
    AutoMLWorkflow,
    PyCaretPipeline,
)


class TestPyCaretPipeline:
    """Test PyCaretPipeline functionality"""

    def test_initialization_without_pycaret(self):
        """Test initialization when PyCaret is not available"""
        with patch("src.core.ml.pycaret_integration.PYCARET_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyCaret is not installed"):
                PyCaretPipeline("classification")

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_initialization_with_pycaret(self):
        """Test initialization when PyCaret is available"""
        pipeline = PyCaretPipeline("classification")
        assert pipeline.task_type == "classification"
        assert pipeline.experiment is not None
        assert pipeline.is_setup is False
        assert pipeline.best_model is None

    def test_unsupported_task_type(self):
        """Test error on unsupported task type"""
        if PYCARET_AVAILABLE:
            with pytest.raises(ValueError, match="Unsupported task type"):
                PyCaretPipeline("unsupported_task")

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_get_available_models_not_setup(self):
        """Test getting available models before setup"""
        pipeline = PyCaretPipeline("classification")
        models = pipeline.get_available_models()
        assert isinstance(models, list)
        # Should return fallback models
        assert "rf" in models or "lr" in models

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_model_info(self):
        """Test getting model information"""
        pipeline = PyCaretPipeline("classification")
        info = pipeline.get_model_info("rf")
        assert isinstance(info, dict)
        assert "name" in info

    def test_setup_without_pycaret(self):
        """Test setup method when PyCaret is not available"""
        with patch("src.core.ml.pycaret_integration.PYCARET_AVAILABLE", False):
            with pytest.raises(ImportError):
                PyCaretPipeline("classification")


class TestAutoMLWorkflow:
    """Test AutoMLWorkflow functionality"""

    def test_initialization_without_pycaret(self):
        """Test initialization when PyCaret is not available"""
        with patch("src.core.ml.pycaret_integration.PYCARET_AVAILABLE", False):
            with pytest.raises(ImportError, match="PyCaret is not installed"):
                AutoMLWorkflow()

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_initialization_with_pycaret(self):
        """Test initialization when PyCaret is available"""
        workflow = AutoMLWorkflow()
        assert workflow.pipeline is None
        assert workflow.results == {}

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_run_automl_mock_success(self, sample_dataframe):
        """Test successful AutoML run with mocked PyCaret"""
        workflow = AutoMLWorkflow()

        # Mock the pipeline methods
        with patch.object(workflow, "pipeline", create=True) as mock_pipeline:
            mock_pipeline.setup = Mock()
            mock_pipeline.compare_models = Mock(return_value=Mock())
            mock_pipeline.tune_hyperparameters = Mock(return_value=Mock())
            mock_pipeline.evaluate_model = Mock(
                return_value={
                    "model": Mock(),
                    "metrics": {"Accuracy": 0.85, "Precision": 0.82},
                    "plots": Mock(),
                }
            )
            mock_pipeline.finalize_model = Mock(return_value=Mock())

            # Create mock pipeline instance
            with patch(
                "src.core.ml.pycaret_integration.PyCaretPipeline"
            ) as MockPipeline:
                MockPipeline.return_value = mock_pipeline

                results = workflow.run_automl(
                    data=sample_dataframe, task_type="classification", target="target"
                )

                assert "pipeline" in results
                assert "best_models" in results
                assert "evaluations" in results
                assert "final_model" in results
                assert results["task_type"] == "classification"
                assert results["target"] == "target"

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_get_predictions_no_model(self):
        """Test getting predictions without a model"""
        workflow = AutoMLWorkflow()
        predictions = workflow.get_predictions()
        assert predictions is None

    @pytest.mark.skipif(not PYCARET_AVAILABLE, reason="PyCaret not installed")
    def test_get_model_interpretation_no_model(self):
        """Test getting interpretation without a model"""
        workflow = AutoMLWorkflow()
        interpretation = workflow.get_model_interpretation()
        assert interpretation is None


class TestPyCaretIntegrationFallbacks:
    """Test fallback behavior when PyCaret is not available"""

    def test_import_safety(self):
        """Test that module can be imported even without PyCaret"""
        # This test passes if the module imports successfully
        from src.core.ml.pycaret_integration import AutoMLWorkflow, PyCaretPipeline

        assert AutoMLWorkflow is not None
        assert PyCaretPipeline is not None

    def test_pycaret_available_flag(self):
        """Test PYCARET_AVAILABLE flag reflects actual availability"""
        from src.core.ml.pycaret_integration import PYCARET_AVAILABLE

        try:
            import pycaret

            assert PYCARET_AVAILABLE is True
        except ImportError:
            assert PYCARET_AVAILABLE is False
