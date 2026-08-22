#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${STREAMLIT_PORT:-8502}"
URL="http://127.0.0.1:${PORT}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then
  echo "STREAMLIT_PORT must be an integer between 1024 and 65535." >&2
  exit 2
fi

if command -v curl >/dev/null 2>&1 && curl --fail --silent "${URL}/_stcore/health" >/dev/null 2>&1; then
  echo "The Streamlit backup is already running at ${URL}"
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "This launcher requires uv: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd "$APP_DIR"
echo "Preparing the isolated Streamlit backup environment..."
uv sync --frozen

echo "Starting the original UncertaintyCat application at ${URL}"
echo "Press Ctrl+C to stop it. Override the port with STREAMLIT_PORT=<port>."
exec uv run --frozen streamlit run UncertaintyCat.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
