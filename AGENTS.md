# AGENTS.md

## Working Rules
- Never commit unless the user explicitly approves it.
- If a change is large or touches many files, propose a short plan and wait for approval before editing.
- Do not create or edit files outside this workspace without explicit permission.
- Do not install dependencies unless explicitly requested.
- Before making code changes, state what will be changed and why.
- Be direct and factual; prioritize correctness over tone-polishing.

## Repository Overview
- Package code: `py/minimint/`
  - `mist_interpolator.py`: MIST track download/preprocessing/interpolation APIs.
  - `bolom.py`: bolometric correction loading/interpolation.
  - `utils.py`: shared interpolation/path helpers.
- Tests: `tests/` (main suite: `tests/test_minimint.py`).
- Example notebook: `examples/Example.ipynb`.
- Generated artifacts: `build/`, `dist/` (do not edit manually).

## Development Commands
- Install editable: `python -m pip install -e .`
- Run tests: `pytest -s`
- Run coverage: `pytest --cov=minimint -s`
- CI lint gate: `pylint -E --disable=E1101 py/minimint/*py`
- Build sdist/wheel: `python -m build --sdist --wheel`

## Style Expectations
- Python style consistent with existing files.
- 4-space indentation.
- `snake_case` for functions/variables; `CapWords` for classes.
- Keep public API names stable (see `py/minimint/__init__.py`).
- Add concise docstrings for non-trivial numerical logic and boundary handling.

## Testing Expectations
- Add/update tests in `tests/test_minimint.py` for behavioral changes.
- Prefer focused regression tests for interpolation and edge cases.
- Use `LOCAL_TESTING=1` for quick local iteration.
- Run full test suite before finalizing significant interpolation/data-path changes.

## Git Hygiene
- Keep changes focused and minimal.
- Never revert unrelated user changes.
- Avoid committing generated or cache artifacts.
- Use short imperative commit titles when commits are requested.

## PR Checklist (when requested)
- What changed and why.
- API/numerical behavior impact.
- Validation commands and results.
- Linked issue(s), if any.
