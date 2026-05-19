# Contributing

Thanks for your interest in signalweave.

## Setup

```bash
git clone https://github.com/alfianmushh-eng/signalweave.git
cd signalweave
pip install -e ".[dev]"
```

## Tests

```bash
pytest
ruff check src tests
```

## Pull requests

- Keep commits small and focused
- Include tests for new features
- Maintain numpy/scipy dependency; avoid heavy frameworks
