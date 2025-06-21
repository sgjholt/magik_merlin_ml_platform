# ML Platform Testing Guide

This directory contains comprehensive tests for the ML Platform application using pytest.

## Test Organization

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── unit/                    # Unit tests (fast, isolated)
│   ├── test_config.py       # Configuration management tests
│   ├── test_data_sources.py # Data source functionality tests
│   └── test_ui_panels.py    # UI component tests
├── integration/             # Integration tests (slower, cross-component)
│   ├── test_app_integration.py    # Full application integration
│   └── test_data_workflow.py      # End-to-end data workflows
└── fixtures/                # Test data and fixtures
```

## Testing Framework

We use **pytest** as our testing framework with the following key features:

- **Fixtures**: Shared test data and setup/teardown
- **Markers**: Categorize tests by type and characteristics
- **Coverage**: Code coverage reporting with pytest-cov
- **Parametrization**: Run tests with multiple inputs

## Test Categories

### Unit Tests
- **Fast execution** (< 1 second each)
- **Isolated components** - test single functions/classes
- **No external dependencies** - mock external services
- **High coverage** - aim for >90% line coverage

### Integration Tests  
- **Cross-component testing** - test component interactions
- **End-to-end workflows** - complete user scenarios
- **Real dependencies** - use actual data sources when possible
- **Slower execution** - may take several seconds

## Running Tests

### All Tests
```bash
# Run all tests
make test

# Or directly with pytest
python -m pytest tests/ -v
```

### Unit Tests Only
```bash
# Fast unit tests only
make test-unit

# Or with pytest
python -m pytest tests/unit/ -v
```

### Integration Tests Only
```bash
# Integration tests only
make test-integration

# Or with pytest
python -m pytest tests/integration/ -v
```

### Coverage Reports
```bash
# Run tests with coverage
make test-coverage

# Or with pytest
python -m pytest tests/ --cov=src --cov-report=html --cov-report=term
```

### Fast Development Testing
```bash
# Run only fast tests (exclude slow/integration)
make test-fast

# Run tests and stop on first failure
python -m pytest tests/unit/ -x --disable-warnings
```

## Test Markers

We use pytest markers to categorize tests:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.slow` - Tests that take >5 seconds
- `@pytest.mark.ui` - Tests involving UI components
- `@pytest.mark.data` - Tests involving data processing

### Running Specific Test Types
```bash
# Run only unit tests
python -m pytest -m unit

# Run only fast tests (exclude slow)
python -m pytest -m "not slow"

# Run only data-related tests
python -m pytest -m data
```

## Test Fixtures

Key fixtures available for all tests:

- `test_data_dir` - Temporary directory with sample data files
- `sample_dataframe` - Sample pandas DataFrame for testing
- `data_source_config` - Basic data source configuration
- `app_instance` - ML Platform app instance (without serving)

## Writing Tests

### Unit Test Example
```python
def test_data_source_creation(data_source_config):
    """Test data source creates correctly"""
    datasource = LocalFileDataSource(data_source_config)
    assert datasource.config == data_source_config
    assert datasource._cache == {}
```

### Integration Test Example
```python
def test_complete_workflow(test_data_dir):
    """Test end-to-end data loading workflow"""
    # Setup
    config = DataSourceConfig(...)
    datasource = LocalFileDataSource(config)
    
    # Test workflow
    assert datasource.test_connection()
    files = datasource.list_tables()
    df = datasource.load_data(files[0])
    
    # Verify results
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
```

## Test Configuration

### pytest.ini
Main pytest configuration including:
- Test discovery patterns
- Coverage settings
- Markers definition
- Warning filters

### Makefile Targets
Convenient test commands:
- `make test` - Run all tests
- `make test-unit` - Unit tests only
- `make test-integration` - Integration tests only
- `make test-coverage` - Tests with coverage
- `make test-fast` - Fast tests for development

## Continuous Integration

Tests are designed to work in CI environments:

1. **Unit tests** run in all CI builds (fast feedback)
2. **Integration tests** run in full CI pipelines
3. **Coverage reports** are generated for all test runs
4. **Test markers** allow selective test execution

## Best Practices

### Test Organization
- One test file per source module
- Group related tests in classes
- Use descriptive test names
- Include docstrings for complex tests

### Test Data
- Use fixtures for test data setup
- Keep test data small and focused
- Clean up temporary files
- Use realistic but minimal data

### Assertions
- One logical assertion per test
- Use pytest's assert statements
- Include descriptive error messages
- Test both success and failure cases

### Mocking
- Mock external dependencies in unit tests
- Use real dependencies in integration tests
- Mock at the appropriate abstraction level
- Verify mock calls when relevant

## Coverage Targets

- **Unit tests**: >95% line coverage
- **Integration tests**: >80% workflow coverage
- **Overall**: >90% combined coverage

Coverage reports are generated in `htmlcov/` directory.

## Debugging Tests

### Verbose Output
```bash
python -m pytest tests/ -v -s
```

### Stop on First Failure
```bash
python -m pytest tests/ -x
```

### Run Specific Test
```bash
python -m pytest tests/unit/test_data_sources.py::TestLocalFileDataSource::test_load_csv_data -v
```

### Debug with PDB
```bash
python -m pytest tests/ --pdb
```

## Performance Considerations

- Unit tests should complete in <1 second each
- Integration tests may take 5-30 seconds
- Use `@pytest.mark.slow` for tests >5 seconds
- Optimize fixtures to avoid repeated setup
- Consider parallel test execution for large test suites