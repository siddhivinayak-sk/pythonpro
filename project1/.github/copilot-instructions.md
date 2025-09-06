# GitHub Copilot Instructions for `pythonpro`

## Project Overview
`pythonpro` is a Python-based project with a focus on data processing, machine learning, and utility scripts. The repository is structured to support modular development, with dedicated directories for specific functionalities such as data handling, AI integrations, and reporting.

### Key Directories
- **`main/`**: Core Python scripts covering various programming concepts and libraries (e.g., NumPy, pandas, matplotlib).
- **`project/`**: Contains configuration utilities and project-specific modules.
- **`tensorflow_and_keras/`**: Scripts for TensorFlow and Keras-based machine learning tasks.
- **`gitlab/`**: Utilities for interacting with GitLab, such as issue and merge request handling.
- **`report/`**: Scripts for generating and testing reports.

### External Dependencies
- **pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **matplotlib**: For data visualization.
- **TensorFlow/Keras**: For machine learning tasks.
- **GitLab API**: For GitLab integrations.

## Developer Workflows

### Setting Up the Environment
1. Install dependencies:
   ```bash
   pip install -r requirements/requirements.txt
   ```
2. For local development, install the package in editable mode:
   ```bash
   python -m pip install -e .
   ```

### Running Scripts
- Execute individual scripts directly, e.g.,
  ```bash
  python main/numpy_usage.py
  ```

### Testing
- Use `tox` for running tests:
  ```bash
  tox
  ```

### Building the Package
- Create a source distribution:
  ```bash
  python -m build --sdist
  ```

## Project-Specific Conventions
- **File Naming**: Use descriptive names that reflect the script's purpose (e.g., `numpy_usage.py`, `exception_handling.py`).
- **Logging**: Use the `logger` module in `project/` for consistent logging.
- **Configuration**: Store environment-specific settings in `project/config/env.py`.

## Examples of Common Patterns

### DataFrame Operations
- Example: Saving a pandas DataFrame as a CSV file:
  ```python
  import pandas as pd

  df = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
  df.to_csv('output.csv', index=False)
  ```

### Exception Handling
- Use `exception_handling.py` for reusable exception handling patterns.

### GitLab Utilities
- Example: Merging JSON files:
  ```bash
  python gitlab/merge_json_files.py
  ```

## Notes for AI Agents
- Follow the coding standards outlined in `.github/instructions/daily.instructions.md`.
- Ensure all new scripts include docstrings and type annotations.
- Validate changes with appropriate tests before committing.

---
This document is a living guide. Update it as the project evolves.
