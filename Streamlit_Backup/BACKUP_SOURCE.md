# Streamlit backup provenance

This directory is a clean source export of the original Streamlit application from:

- repository: <https://github.com/MLegkovskis/UncertaintyCat>
- branch: `main_bk_2026_Aug`
- source commit: `f1fe78202ffcd0252a7e079bd99b88837e47293f`
- source commit date: `2026-01-13T18:23:29+00:00`

The source tree is unchanged from that commit except for this provenance file and
`start_streamlit.sh`, which were added on the modern application's `main` branch for local reference.
It deliberately retains the backup's OpenTURNS 1.25 dependency and runs in its own `.venv`, so it does not
alter the modern application's OpenTURNS 1.27 environment.

## Start the backup

From the modern UncertaintyCat repository root:

```bash
./Streamlit_Backup/start_streamlit.sh
```

Then open <http://127.0.0.1:8502>. Stop the foreground server with `Ctrl+C`.
Running the launcher again automatically stops the previous backup instance first, ensuring a clean launch.
It does not stop unrelated Streamlit applications.

To use another port:

```bash
STREAMLIT_PORT=8510 ./Streamlit_Backup/start_streamlit.sh
```

The numerical features run without an AI key. The original AI commentary controls still expect the legacy
provider configuration described in `readme.md`; no legacy AI secret is bundled or inherited from production.

## Archived example check

The branch's example harness imports `colorama`, although the historical lockfile does not declare it. Run the
check with a temporary dependency overlay, leaving the snapshot and its environment unchanged:

```bash
cd Streamlit_Backup
uv run --frozen --with colorama python test_all_examples.py
```

All 23 archived examples passed this check when the backup was imported.
