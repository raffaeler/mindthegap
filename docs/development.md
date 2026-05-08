# Development

## Test, lint, and type-check

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Notes

- `tests/` covers cache, proxy behavior, streaming, TLS, and transforms.
- `src/mindthegap/` contains the app, CLI, TLS handling, streaming logic, and
  request/response transforms.