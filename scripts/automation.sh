#!/usr/bin/env bash

set -euo pipefail

repo="$(gh repo view --json nameWithOwner --jq .nameWithOwner)"

usage() {
  echo "Usage: $0 {pause|resume|status|release}" >&2
  exit 2
}

status() {
  local value
  value="$(gh variable list --repo "$repo" --json name,value --jq '.[] | select(.name == "AUTOMATION_ENABLED") | .value')"
  if [[ -z "$value" ]]; then
    value="unset (automatic workflows remain disabled)"
  fi
  printf 'AUTOMATION_ENABLED=%s\n' "$value"
}

case "${1:-}" in
  pause)
    gh variable set AUTOMATION_ENABLED --body false --repo "$repo"
    status
    ;;
  resume)
    gh variable set AUTOMATION_ENABLED --body true --repo "$repo"
    status
    ;;
  status)
    status
    ;;
  release)
    if [[ -n "$(git status --porcelain)" ]]; then
      echo "Release refused: the working tree is not clean." >&2
      exit 1
    fi
    branch="$(git branch --show-current)"
    if [[ "$branch" != "main" ]]; then
      echo "Release refused: current branch is '$branch', expected 'main'." >&2
      exit 1
    fi
    git fetch --quiet origin main
    local_sha="$(git rev-parse HEAD)"
    remote_sha="$(git rev-parse origin/main)"
    if [[ "$local_sha" != "$remote_sha" ]]; then
      echo "Release refused: local main does not equal origin/main." >&2
      exit 1
    fi
    gh workflow run ci.yml --repo "$repo" --ref main
    run_url=""
    for _ in {1..20}; do
      run_url="$(gh run list --repo "$repo" --workflow ci.yml --commit "$local_sha" --event workflow_dispatch --limit 1 --json url --jq '.[0].url // empty')"
      [[ -n "$run_url" ]] && break
      sleep 1
    done
    if [[ -z "$run_url" ]]; then
      echo "CI was dispatched for $local_sha, but its URL is not available yet." >&2
      exit 1
    fi
    printf 'Release CI: %s\n' "$run_url"
    ;;
  *)
    usage
    ;;
esac
