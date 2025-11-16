# Code Quality Standards

This document explains our code quality standards, linting rules, and the rationale behind our configuration choices.

## Overview

We use strict code quality tools to maintain consistency and catch bugs early:

- **Ruff**: Fast linter and formatter (replaces Black + Flake8 + isort)
- **MyPy**: Static type checking
- **Pytest**: Testing framework with coverage tracking

## Linting Configuration

### Global Standards

We enable a comprehensive set of Ruff rules (see `pyproject.toml` for full list):

```toml
[tool.ruff.lint]
select = [
    "E", "W",    # pycodestyle
    "F",         # pyflakes
    "B",         # flake8-bugbear
    "I",         # isort
    "N",         # pep8-naming
    "UP",        # pyupgrade
    "ANN",       # flake8-annotations
    # ... many more
]
```

### Ignored Rules and Why

Some rules are ignored globally because they conflict with our goals or are overly strict for ML code.

#### Formatter Conflicts

```toml
"E501",    # Line too long (handled by formatter)
"COM812",  # Trailing comma missing
"ISC001",  # Implicit string concatenation
```

**Why**: These rules conflict with Ruff's auto-formatter. The formatter handles these issues automatically.

#### Python Conventions

```toml
"ANN101",  # Missing type annotation for self
"ANN102",  # Missing type annotation for cls
```

**Why**: Type annotations for `self` and `cls` are implicit in Python - adding them is redundant and reduces readability.

#### Practical Considerations

```toml
"T201",    # Print statements
```

**Why**: Print statements are acceptable for:
- CLI output and logging
- Debugging during development
- Example scripts and demos

```toml
"S101",    # Use of assert
```

**Why**: Assert statements are standard practice in pytest tests and useful for precondition checks.

#### ML/Scientific Computing

```toml
"FBT001",  # Boolean positional arg
"FBT002",  # Boolean default value
"PLR0913", # Too many arguments
"PLR0912", # Too many branches
"PLR0915", # Too many statements
```

**Why**: ML models often need:
- Many hyperparameters (e.g., `n_estimators`, `max_depth`, `learning_rate`)
- Boolean flags for different modes (e.g., `fit_intercept`, `normalize`)
- Complex model selection logic

Being overly strict here would hurt usability and sklearn compatibility.

## ML-Specific Rules

### The `X` and `y` Convention

The most important ML-specific rules:

```toml
"src/core/ml_engine/**/*.py" = [
    "N803",  # Argument name should be lowercase
    "N806",  # Variable should be lowercase
]
```

**Why This Matters:**

In machine learning, **`X` (uppercase)** for features and **`y` (lowercase)** for targets is the **universal convention**:

```python
# Standard ML code (used everywhere)
model.fit(X, y)
predictions = model.predict(X)

# What ruff wants (confusing and non-standard)
model.fit(x, y)  # ❌ Violates ML conventions
```

This convention comes from:
- **Mathematics notation**: X is a matrix (uppercase), y is a vector (lowercase)
- **Every ML library**: sklearn, pandas, numpy, TensorFlow, PyTorch
- **All ML tutorials**: Coursera, fast.ai, Kaggle, research papers

**Following strict PEP8 naming here would make our code confusing to ML practitioners.**

### Type Flexibility

```toml
"ANN401",  # Dynamically typed expressions (typing.Any)
```

**Why**: ML functions must accept multiple input types:

```python
def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
    # Could also be: List[List[float]], scipy.sparse.csr_matrix, etc.
    # Using Any for internal helpers is more practical than complex unions
```

Alternatives like `Union[pd.DataFrame, np.ndarray, List[List[float]], ...]` become unmanageable.

### Import Organization

```toml
"TID252",  # Prefer absolute imports over relative
```

**Why**: Within the `ml_engine` package, relative imports are cleaner:

```python
# Relative (cleaner, easier to refactor)
from .base import BaseClassifier
from .classical import XGBoostClassifier

# Absolute (verbose, harder to maintain)
from src.core.ml_engine.base import BaseClassifier
from src.core.ml_engine.classical import XGBoostClassifier
```

### Logging Style

```toml
"G004",  # f-strings in logging
```

**Why**: Modern Python (3.13+) and readability:

```python
# Modern, readable
logger.info(f"Training {model_name} with {n_samples} samples")

# Old style (what ruff prefers)
logger.info("Training %s with %d samples", model_name, n_samples)
```

F-strings are faster in Python 3.13+ and more readable. Performance difference is negligible for logging.

### Magic Numbers

```toml
"PLR2004",  # Magic value used in comparison
```

**Why**: In ML, certain values are domain knowledge, not "magic":

```python
if n_classes > 2:  # Multiclass vs binary
    objective = "multi:softprob"
else:
    objective = "binary:logistic"
```

Creating constants like `BINARY_THRESHOLD = 2` adds verbosity without adding clarity.

### Type-Checking Blocks

```toml
"TC001",  # Move application import into type-checking block
"TC002",  # Move third-party import into type-checking block
```

**Why**: Our imports are used at runtime, not just for type hints:

```python
# We need these at runtime
import numpy as np
from ..experiments.tracking import ExperimentTracker

# Not just for type hints
def compare_models(self, X: np.ndarray, tracker: ExperimentTracker):
    # Actually using numpy and tracker here
    results = np.array([...])
    tracker.log_metrics(results)
```

## Test-Specific Rules

```toml
"tests/**/*.py" = [
    "S101",    # Use of assert (standard in pytest)
    "ANN",     # Type annotations not required
    "PLR2004", # Magic values acceptable
    "SLF001",  # Private member access needed
]
```

**Why**:
- **Assertions** are the foundation of pytest
- **Type annotations** in tests add noise without value
- **Magic values** in test assertions are self-documenting (e.g., `assert len(results) == 10`)
- **Private access** necessary to test internal implementation

## Running Quality Checks

### Linting

```bash
# Check all code
uv run ruff check src/ tests/

# Auto-fix issues
uv run ruff check --fix src/ tests/

# Check formatting
uv run ruff format --check src/ tests/

# Auto-format
uv run ruff format src/ tests/
```

### Type Checking

```bash
# Run mypy
uv run mypy src/
```

### All Checks

```bash
# Run all quality checks
./run.sh lint
```

## Pre-Commit Hooks

We use pre-commit hooks to enforce quality automatically:

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks will:
1. Format code with ruff
2. Check for linting errors
3. Run type checks
4. Verify no debug statements

## CI/CD Integration

GitHub Actions runs all quality checks on every push:

```yaml
- name: Run ruff linting
  run: uv run ruff check src/ tests/

- name: Run ruff formatting check
  run: uv run ruff format --check src/ tests/
```

Pull requests cannot be merged if quality checks fail.

## When to Add Exceptions

### Adding `# noqa` Comments

For rare cases where a rule doesn't apply:

```python
def get_params(self, deep: bool = True):  # noqa: ARG002
    # 'deep' required by sklearn interface but not used
    return self.params.copy()
```

### Adding Per-File Ignores

For systematic exceptions in specific files:

```toml
"src/new_module/**/*.py" = [
    "RULE",  # Explanation of why this rule doesn't apply
]
```

### Proposing Global Changes

If a rule consistently causes issues:
1. Document the problem with examples
2. Discuss in a GitHub issue
3. Update this documentation
4. Update `pyproject.toml`

## Best Practices

### Writing Clean Code

1. **Follow conventions**: Use established patterns from sklearn, pandas
2. **Document decisions**: Add docstrings explaining "why" not just "what"
3. **Test thoroughly**: Write tests before disabling rules
4. **Think about users**: Would an ML practitioner understand this?

### Handling Linting Errors

When ruff reports an error:

1. **Understand why** the rule exists (check this doc)
2. **Fix the issue** if the rule makes sense
3. **Add `# noqa`** if the rule doesn't apply in this specific case
4. **Propose changes** if the rule doesn't fit our project

### Examples

#### Good: Following ML Conventions

```python
def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
    """Fit model using standard ML interface."""
    X = self._validate_input(X)  # X uppercase is correct
    self.model.fit(X, y)
    return self
```

#### Bad: Fighting the Linter

```python
def fit(self, x: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray):
    """Fit model."""  # Using lowercase x confuses ML developers
    x_data = self._validate_input(x)
    self.model.fit(x_data, y)
    return self
```

## Summary

Our code quality standards balance:

- ✅ **Strict standards** for maintainability and bug prevention
- ✅ **ML conventions** for usability and clarity
- ✅ **Modern Python** (3.13+) features and patterns
- ✅ **Practical exceptions** where rules hurt more than help

When in doubt, prioritize **code clarity** and **ML conventions** over strict PEP8 compliance.

## References

- [Ruff Documentation](https://docs.astral.sh/ruff/)
- [Sklearn Development Guide](https://scikit-learn.org/stable/developers/develop.html)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
