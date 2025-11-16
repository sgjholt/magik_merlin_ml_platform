# GitHub Workflows Templates

This directory contains GitHub Actions workflow templates that can be used for CI/CD.

## Setup Instructions

To enable automated testing with GitHub Actions:

1. Create the `.github/workflows/` directory in your repository:
   ```bash
   mkdir -p .github/workflows
   ```

2. Copy the workflow file:
   ```bash
   cp .github-templates/workflows/ci.yml .github/workflows/ci.yml
   ```

3. Commit and push to enable GitHub Actions:
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "chore: Enable GitHub Actions CI/CD"
   git push
   ```

## What the CI/CD Pipeline Does

The `ci.yml` workflow provides:

- **Automated Testing**: Runs unit and integration tests on every push/PR
- **Code Quality**: Linting and formatting checks with ruff
- **Coverage Reports**: Generates and uploads test coverage
- **Build Validation**: Ensures the package builds correctly
- **Python 3.12 Support**: Tests on the latest Python version

## Workflow Jobs

1. **test**: Runs all tests with coverage reporting
2. **lint**: Checks code quality with ruff
3. **build**: Validates package can be built

## Triggers

The workflow runs on:
- Pushes to: `main`, `master`, `develop`, `claude/**` branches
- Pull requests to: `main`, `master`, `develop` branches
