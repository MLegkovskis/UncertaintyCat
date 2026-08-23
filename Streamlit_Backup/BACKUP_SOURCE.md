# Streamlit backup provenance

This directory is a clean source export of the original Streamlit application from:

- repository: <https://github.com/MLegkovskis/UncertaintyCat>
- branch: `main_bk_2026_Aug`
- source commit: `f1fe78202ffcd0252a7e079bd99b88837e47293f`
- source commit date: `2026-01-13T18:23:29+00:00`

The archived source tree is unchanged from that commit except for this provenance file. It deliberately
retains the backup's OpenTURNS 1.25 dependency and lockfile, but it is excluded from the modern package,
CI, and deployment graphs.

## Reference-only policy

This folder exists only for historical behavior and prompt comparison. Do not import it from modern code,
add it to root dependencies, patch it as part of product work, or create repository launch scripts for it.
New functionality belongs in `uncertaintycat_core`, `services/compute`, `apps/api`, or `apps/web`.

## Archived example check

The branch's example harness imports `colorama`, although the historical lockfile does not declare it. Run the
check with a temporary dependency overlay, leaving the snapshot and its environment unchanged:

```bash
cd Streamlit_Backup
uv run --frozen --with colorama python test_all_examples.py
```

All 23 archived examples passed this check when the backup was imported. This is historical evidence, not
part of the modern release gate.
