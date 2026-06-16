# Development

## Test, lint, and type-check

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy src
```

## Notes

- `tests/` covers cache, proxy behavior, streaming, TLS, transforms, upstream
  resolution, and reasoning text sanitisation.
- `src/mindthegap/` contains the app, CLI, TLS handling, streaming logic,
  request/response transforms, upstream resolution, and DSML/XML sanitisation.