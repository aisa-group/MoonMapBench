#!/usr/bin/env bash
# Query a model on the lat/lon grid and score it against LOLA.
# Usage: scripts/run.sh [--res 22.5] [--samples 1] [--model gpt-6.1-sol] [--effort low] [--concurrency 8] [--timeout 600]
# Cached answers in results/<model>__<effort>/responses.jsonl are reused.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/venvs/moonmapbench"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY to an OpenAI API key}"

query_args=()
report_args=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --res|--samples|--model|--effort) query_args+=("$1" "$2"); report_args+=("$1" "$2"); shift 2 ;;
        --concurrency|--timeout) query_args+=("$1" "$2"); shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ${a[@]+"${a[@]}"}: empty arrays are "unbound" under set -u in bash < 4.4 (macOS)
uv run python -m moonmapbench.query ${query_args[@]+"${query_args[@]}"}
uv run python -m moonmapbench.report ${report_args[@]+"${report_args[@]}"}
