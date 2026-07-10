# Validation Report

Status: pass

Checks run:

- `python -m py_compile tpi_jepa/*.py scripts/*.py`: pass
- `pytest tests`: collection failed because the console entry point did not put
  the repo root on `sys.path`; imports for `tpi_jepa` and `scripts` failed.
- `python -m pytest tests`: pass, 9 tests passed.
- local Markdown link scan for `docs/operator_runbook.md` and
  `docs/script_inventory.md`: pass, no Markdown file links found.
- common secret-pattern scan over generated docs and learn artifacts: pass.

No code changes were required by this learn run.
