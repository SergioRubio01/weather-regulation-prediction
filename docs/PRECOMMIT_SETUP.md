# Pre-commit Setup Guide

This project uses pre-commit hooks to maintain code quality and consistency. Pre-commit runs various checks on your code before each commit.

## Installation

1. **Install pre-commit** (if not already installed):

   ```bash
   # Using pip
   pip install pre-commit

   # Or if using Poetry
   poetry add --group dev pre-commit
   ```

2. **Install the git hook scripts**:

   ```bash
   pre-commit install
   ```

3. **Run against all files** (optional, but recommended for first setup):

   ```bash
   pre-commit run --all-files
   ```

## What's Included

Our pre-commit configuration includes the following tools:

### Code Formatters

- **Black**: Python code formatter (line length: 100)
- **isort**: Import statement sorter
- **docformatter**: Docstring formatter
- **autoflake**: Removes unused imports and variables
- **pyupgrade**: Automatically upgrades Python syntax

### Linters

- **Ruff**: Fast Python linter
- **Flake8**: Style guide enforcement with plugins:
  - flake8-docstrings
  - flake8-comprehensions
  - flake8-bugbear
  - flake8-simplify
- **Pylint**: Python code analyzer
- **mypy**: Static type checker
- **pydocstyle**: Docstring style checker

### Security

- **Bandit**: Security issue scanner
- **detect-secrets**: Prevents secrets from being committed

### Other Tools

- **YAML lint**: YAML file validation
- **Markdown lint**: Markdown file validation
- **JSON formatter**: Formats and validates JSON files
- **Jupyter notebook cleanup**: Removes output from notebooks
- **General hooks**: Trailing whitespace, file size limits, merge conflicts, etc.

## Usage

### Automatic Checks

Once installed, pre-commit will automatically run on `git commit`. If any checks fail:

1. The commit will be aborted
2. Files will be auto-fixed where possible
3. You'll need to review changes and commit again

### Manual Usage

```bash
# Run on all files
pre-commit run --all-files

# Run on specific files
pre-commit run --files file1.py file2.py

# Run specific hook
pre-commit run black --all-files

# Update hooks to latest versions
pre-commit autoupdate
```

### Skipping Hooks

If you need to skip pre-commit for a specific commit:

```bash
git commit --no-verify -m "Your commit message"
```

**Note**: Use this sparingly and only when absolutely necessary.

## Configuration

The configuration is stored in `.pre-commit-config.yaml`. Tool-specific settings are in:

- `pyproject.toml`: Black, isort, Ruff, mypy, pylint, pytest, coverage
- `.flake8`: Flake8 configuration
- `.yamllint`: YAML linting rules
- `.markdownlint.json`: Markdown linting rules
- `.secrets.baseline`: Baseline for secret detection

## Troubleshooting

### Black and isort conflicts

Both tools are configured to work together with compatible settings.

### Large file warnings

Files larger than 1MB will be flagged. If you need to commit large files, consider:

- Using Git LFS
- Excluding them from version control
- Adjusting the limit in `.pre-commit-config.yaml`

### Performance

The first run may be slow as it sets up environments. Subsequent runs are much faster.

### Poetry lock file

The poetry-lock hook ensures `poetry.lock` stays in sync with `pyproject.toml`.

## Continuous Integration

Pre-commit can also run in CI. The configuration includes settings for pre-commit.ci, which automatically:

- Runs on pull requests
- Auto-fixes issues where possible
- Updates hooks weekly
