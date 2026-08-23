#!/usr/bin/env bash

set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
state_dir="$root_dir/.wrangler/local-stack"
pid_file="$state_dir/processes"
log_dir="$state_dir/logs"
wrangler="$root_dir/apps/api/node_modules/wrangler/bin/wrangler.js"
wrangler_runtime=(node "$wrangler")
wrangler_config="apps/api/wrangler.no-ai.jsonc"
ai_status="unavailable (Wrangler is not authenticated)"

mkdir -p "$log_dir"

stop_recorded_processes() {
  [[ -f "$pid_file" ]] || return 0
  while read -r service pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    [[ -d "/proc/$pid" ]] || continue
    process_cwd="$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)"
    process_command="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
    if [[ "$process_cwd" == "$root_dir"* ]] &&
      [[ "$process_command" == *uncertaintycat* ||
        "$process_command" == *uvicorn* ||
        "$process_command" == *wrangler* ||
        "$process_command" == *vite* ]]; then
      kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
  done < "$pid_file"
  : > "$pid_file"
}

children=()
cleanup() {
  trap - EXIT INT TERM
  for pid in "${children[@]}"; do
    kill -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  : > "$pid_file"
}

start_service() {
  local name="$1"
  shift
  setsid "$@" >"$log_dir/$name.log" 2>&1 &
  local pid=$!
  children+=("$pid")
  printf '%s %s\n' "$name" "$pid" >> "$pid_file"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  for _ in {1..120}; do
    if curl --fail --silent "$url" >/dev/null 2>&1; then
      printf '  %-10s ready at %s\n' "$name" "$url"
      return 0
    fi
    sleep 1
  done
  printf '%s did not become healthy. See %s/%s.log\n' "$name" "$log_dir" "$name" >&2
  return 1
}

cd "$root_dir"
stop_recorded_processes
trap cleanup EXIT INT TERM

printf 'Synchronising Python and Node dependencies…\n'
uv sync --frozen --extra dev
npm ci

node_major="$(node -p 'process.versions.node.split(".")[0]')"
if (( node_major < 22 )); then
  printf 'Using an isolated Node 22 runtime for Wrangler (local Node is %s).\n' "$(node --version)"
  wrangler_runtime=(npx --yes node@22 "$wrangler")
fi

if "${wrangler_runtime[@]}" whoami >/dev/null 2>&1; then
  wrangler_config="apps/api/wrangler.jsonc"
  ai_status="remote Workers AI (usage may be billed by Cloudflare)"
fi

if [[ ! -f apps/api/.dev.vars ]]; then
  install -m 600 apps/api/.dev.vars.example apps/api/.dev.vars
fi

printf 'Applying forward-only migrations to isolated local D1…\n'
CI=true "${wrangler_runtime[@]}" d1 migrations apply uncertaintycat-local \
  --local \
  --persist-to "$state_dir/data" \
  --config "$wrangler_config"

: > "$pid_file"
printf 'Starting the retained local workspace…\n'
start_service compute uv run uvicorn services.compute.main:app \
  --host 127.0.0.1 --port 8080
start_service worker "${wrangler_runtime[@]}" dev --local \
  --port 8787 \
  --persist-to "$state_dir/data" \
  --config "$wrangler_config"
start_service web npm run dev --workspace @uncertaintycat/web -- --port 5173

wait_for_url compute http://127.0.0.1:8080/health
wait_for_url worker http://127.0.0.1:8787/health
wait_for_url web http://127.0.0.1:5173

printf '\nUncertaintyCat is ready: http://127.0.0.1:5173\n'
printf 'Identity: Local retained user (DEV_AUTH_BYPASS=true)\n'
printf 'Data: local-only D1/R2/Queues under %s/data\n' "$state_dir"
printf 'Workers AI: %s\n' "$ai_status"
printf 'Press Ctrl+C to stop only this local stack.\n'

wait
