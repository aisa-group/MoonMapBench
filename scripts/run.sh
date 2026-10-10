#!/usr/bin/env bash
# Query a model on the lat/lon grid, score it against LOLA and print the tokens and cost of all saved answers.
# Usage: scripts/run.sh [--res 22.5] [--samples 1] [--model gpt-6.1-sol] [--effort low] [--concurrency 256] [--timeout 600] [--points N]
# Claude models (claude-*) need ANTHROPIC_API_KEY, others OPENAI_API_KEY, in the environment or in .env.
# Cached answers in results/<model>__<effort>/responses.jsonl are reused.
# --points N queries only N grid points, picked at random with a fixed seed (a cheap test), and skips the report.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_PROJECT_ENVIRONMENT="$HOME/.local/share/venvs/moonmapbench"

query_args=()
report_args=()
cost_args=()
partial=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --res|--samples) query_args+=("$1" "$2"); report_args+=("$1" "$2"); shift 2 ;;
        --model|--effort) query_args+=("$1" "$2"); report_args+=("$1" "$2"); cost_args+=("$1" "$2"); shift 2 ;;
        --concurrency|--timeout) query_args+=("$1" "$2"); shift 2 ;;
        --points) query_args+=("$1" "$2"); partial=1; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# ${a[@]+"${a[@]}"}: empty arrays are "unbound" under set -u in bash < 4.4 (macOS)
uv run python -m moonmapbench.query ${query_args[@]+"${query_args[@]}"}
if [[ $partial == 1 ]]; then
    echo "--points: only part of the grid was queried, so there is no report"
else
    uv run python -m moonmapbench.report ${report_args[@]+"${report_args[@]}"}
fi
uv run python -m moonmapbench.cost ${cost_args[@]+"${cost_args[@]}"}
