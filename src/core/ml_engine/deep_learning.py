"""
Deep Learning models using PyTorch Lightning.

This module provides sklearn-compatible wrappers for PyTorch Lightning models,
enabling seamless integration with the ML platform's AutoML pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseClassifier, BaseRegressor

if TYPE_CHECKING:
    import pandas as pd
    from numpy.typing import NDArray


class TabularNet(LightningModule):
    """
    Simple feedforward neural network for tabular data.

    This is a configurable MLP architecture that can be used for both
    classification and regression tasks.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        task_type: str = "classification",
    ) -> None:
        """
        Initialize TabularNet.

        Args:
            input_dim: Number of input features
            output_dim: Number of output classes (classification) or 1 (regression)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            task_type: 'classification' or 'regression'
        """
        super().__init__()
        self.save_hyperparameters()

        if hidden_dims is None:
            hidden_dims = [128, 64]

        # Build network layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)
        self.task_type = task_type

        # Loss function
        if task_type == "classification":
            if output_dim == 1:
                self.loss_fn = nn.BCEWithLogitsLoss()
            else:
                self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(x)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""
        x, y = batch
        y_hat = self(x)

        if self.task_type == "classification" and y_hat.shape[1] > 1:
            loss = self.loss_fn(y_hat, y.long())
        else:
            loss = self.loss_fn(y_hat.squeeze(), y.float())

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        """Validation step."""
        x, y = batch
        y_hat = self(x)

        if self.task_type == "classification" and y_hat.shape[1] > 1:
            loss = self.loss_fn(y_hat, y.long())
            preds = torch.argmax(y_hat, dim=1)
            acc = (preds == y).float().mean()
            self.log("val_acc", acc, prog_bar=True)
        else:
            loss = self.loss_fn(y_hat.squeeze(), y.float())

        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class LightningClassifier(BaseClassifier):
    """
    PyTorch Lightning classifier with sklearn-compatible interface.

    This wrapper provides a familiar sklearn API for PyTorch Lightning models,
    making them compatible with the AutoML pipeline.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        random_state: int | None = 42,
        **trainer_kwargs: Any,
    ) -> None:
        """
        Initialize Lightning classifier.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            random_state: Random seed
            **trainer_kwargs: Additional kwargs for Lightning Trainer
        """
        super().__init__()
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.trainer_kwargs = trainer_kwargs

        self.model_: TabularNet | None = None
        self.trainer_: Trainer | None = None
        self.n_classes_: int | None = None
        self.n_features_in_: int | None = None

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.floating],
        y: pd.Series | NDArray[np.integer],
    ) -> LightningClassifier:
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training targets

        Returns:
            self
        """
        # Convert to numpy arrays
        X_array = X.values if hasattr(X, "values") else X
        y_array = y.values if hasattr(y, "values") else y

        self.n_features_in_ = X_array.shape[1]
        self.classes_ = np.unique(y_array)
        self.n_classes_ = len(self.classes_)

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_array)
        y_tensor = torch.LongTensor(y_array)

        # Create train/validation split
        n_samples = len(X_tensor)
        n_val = int(n_samples * self.validation_split)

        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Create model
        output_dim = self.n_classes_ if self.n_classes_ > 2 else 1  # noqa: PLR2004
        self.model_ = TabularNet(
            input_dim=self.n_features_in_,
            output_dim=output_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            task_type="classification",
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]

        # Create trainer
        self.trainer_ = Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            **self.trainer_kwargs,
        )

        # Train model
        self.trainer_.fit(self.model_, train_loader, val_loader)

        return self

    def predict(self, X: pd.DataFrame | NDArray[np.floating]) -> NDArray[np.integer]:
        """
        Predict class labels.

        Args:
            X: Features to predict

        Returns:
            Predicted class labels
        """
        if self.model_ is None:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        # Convert to tensor
        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.FloatTensor(X_array)

        # Predict
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)

            if self.n_classes_ > 2:  # noqa: PLR2004
                predictions = torch.argmax(logits, dim=1).numpy()
            else:
                predictions = (torch.sigmoid(logits.squeeze()) > 0.5).long().numpy()

        return predictions

    def predict_proba(
        self, X: pd.DataFrame | NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """
        Predict class probabilities.

        Args:
            X: Features to predict

        Returns:
            Class probabilities
        """
        if self.model_ is None:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        # Convert to tensor
        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.FloatTensor(X_array)

        # Predict
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)

            if self.n_classes_ > 2:  # noqa: PLR2004
                probas = torch.softmax(logits, dim=1).numpy()
            else:
                pos_proba = torch.sigmoid(logits.squeeze()).numpy()
                probas = np.column_stack([1 - pos_proba, pos_proba])

        return probas

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> LightningClassifier:
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


class LightningRegressor(BaseRegressor):
    """
    PyTorch Lightning regressor with sklearn-compatible interface.

    This wrapper provides a familiar sklearn API for PyTorch Lightning models,
    making them compatible with the AutoML pipeline.
    """

    def __init__(
        self,
        hidden_dims: list[int] | None = None,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        early_stopping_patience: int = 10,
        random_state: int | None = 42,
        **trainer_kwargs: Any,
    ) -> None:
        """
        Initialize Lightning regressor.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Learning rate for optimizer
            max_epochs: Maximum number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
            early_stopping_patience: Patience for early stopping
            random_state: Random seed
            **trainer_kwargs: Additional kwargs for Lightning Trainer
        """
        super().__init__()
        self.hidden_dims = hidden_dims or [128, 64]
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.trainer_kwargs = trainer_kwargs

        self.model_: TabularNet | None = None
        self.trainer_: Trainer | None = None
        self.n_features_in_: int | None = None

        # Set random seed
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

    def fit(
        self,
        X: pd.DataFrame | NDArray[np.floating],
        y: pd.Series | NDArray[np.floating],
    ) -> LightningRegressor:
        """
        Fit the model to training data.

        Args:
            X: Training features
            y: Training targets

        Returns:
            self
        """
        # Convert to numpy arrays
        X_array = X.values if hasattr(X, "values") else X
        y_array = y.values if hasattr(y, "values") else y

        self.n_features_in_ = X_array.shape[1]

        # Convert to tensors
        X_tensor = torch.FloatTensor(X_array)
        y_tensor = torch.FloatTensor(y_array)

        # Create train/validation split
        n_samples = len(X_tensor)
        n_val = int(n_samples * self.validation_split)

        indices = torch.randperm(n_samples)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

        train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
        val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Create model
        self.model_ = TabularNet(
            input_dim=self.n_features_in_,
            output_dim=1,  # Single output for regression
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            task_type="regression",
        )

        # Setup callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.early_stopping_patience,
                mode="min",
            ),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                save_top_k=1,
            ),
        ]

        # Create trainer
        self.trainer_ = Trainer(
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            **self.trainer_kwargs,
        )

        # Train model
        self.trainer_.fit(self.model_, train_loader, val_loader)

        return self

    def predict(self, X: pd.DataFrame | NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Predict target values.

        Args:
            X: Features to predict

        Returns:
            Predicted values
        """
        if self.model_ is None:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        # Convert to tensor
        X_array = X.values if hasattr(X, "values") else X
        X_tensor = torch.FloatTensor(X_array)

        # Predict
        self.model_.eval()
        with torch.no_grad():
            predictions = self.model_(X_tensor).squeeze().numpy()

        return predictions

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "validation_split": self.validation_split,
            "early_stopping_patience": self.early_stopping_patience,
            "random_state": self.random_state,
        }

    def set_params(self, **params: Any) -> LightningRegressor:
        """Set model parameters."""
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Register Lightning models in the global registry
try:
    from .base import model_registry

    model_registry.register("lightning_classifier", LightningClassifier, "deep_learning")
    model_registry.register("lightning_regressor", LightningRegressor, "deep_learning")
except ImportError:
    pass  # Registry not available during import
