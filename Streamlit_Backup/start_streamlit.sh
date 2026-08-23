#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PORT="${STREAMLIT_PORT:-8502}"
URL="http://127.0.0.1:${PORT}"

if [[ ! "$PORT" =~ ^[0-9]+$ ]] || ((PORT < 1024 || PORT > 65535)); then
  echo "STREAMLIT_PORT must be an integer between 1024 and 65535." >&2
  exit 2
fi

find_previous_instances() {
  local proc_dir pid process_cwd command_line

  for proc_dir in /proc/[0-9]*; do
    pid="${proc_dir##*/}"
    [[ "$pid" == "$$" || "$pid" == "$PPID" ]] && continue
    [[ -r "${proc_dir}/cmdline" ]] || continue

    process_cwd="$(readlink "${proc_dir}/cwd" 2>/dev/null || true)"
    [[ "$process_cwd" == "$APP_DIR" ]] || continue

    command_line="$(tr '\0' ' ' < "${proc_dir}/cmdline" 2>/dev/null || true)"
    if [[ "$command_line" == *streamlit* && "$command_line" == *UncertaintyCat.py* ]]; then
      printf '%s\n' "$pid"
    fi
  done
}

mapfile -t previous_pids < <(find_previous_instances)
if ((${#previous_pids[@]} > 0)); then
  echo "Stopping the previous Streamlit backup instance (${previous_pids[*]})..."
  kill -TERM "${previous_pids[@]}" 2>/dev/null || true

  for _ in {1..50}; do
    mapfile -t remaining_pids < <(find_previous_instances)
    ((${#remaining_pids[@]} == 0)) && break
    sleep 0.1
  done

  mapfile -t remaining_pids < <(find_previous_instances)
  if ((${#remaining_pids[@]} > 0)); then
    echo "Forcing the previous instance to stop (${remaining_pids[*]})..."
    kill -KILL "${remaining_pids[@]}" 2>/dev/null || true

    for _ in {1..20}; do
      mapfile -t remaining_pids < <(find_previous_instances)
      ((${#remaining_pids[@]} == 0)) && break
      sleep 0.1
    done
  fi
fi

if command -v ss >/dev/null 2>&1 && [[ -n "$(ss -H -ltn "sport = :${PORT}")" ]]; then
  echo "Port ${PORT} is occupied by an unrelated process; refusing to stop it." >&2
  echo "Choose another port with STREAMLIT_PORT=<port>." >&2
  exit 1
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
exec uv run --frozen python -m streamlit run UncertaintyCat.py \
  --server.address 127.0.0.1 \
  --server.port "$PORT" \
  --server.headless true \
  --browser.gatherUsageStats false
