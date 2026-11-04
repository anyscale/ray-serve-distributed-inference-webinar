#!/usr/bin/env bash
# Launch Claude Code against the GLM-5.3 Ray Serve LLM service.
# Adapted from anyscale/llm-serving-for-coding-agents:
#   part2-connect-clients-production/claude-service.sh
#
# Required environment (never typed or echoed on screen):
#   GLM_BASE_URL  service root, WITHOUT /v1; Claude Code appends /v1/messages itself
#   GLM_API_KEY   the service's bearer token
# Optional:
#   GLM_MODEL                  served model id (default glm-5.3-nvfp4)
#   WEBINAR_CLAUDE_CONFIG_DIR  clean Claude Code config dir (default ~/.claude-webinar),
#                              so no personal CLAUDE.md, memory or MCP servers load
#
# Usage:
#   ./claude-glm.sh            # fresh copy of demo-bug/ in a temp dir, then claude
#   ./claude-glm.sh <dir>      # run claude in <dir> instead
#   ./claude-glm.sh --check    # preflight only: is the model served and the token valid?
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${GLM_MODEL:-glm-5.3-nvfp4}"

: "${GLM_BASE_URL:?set GLM_BASE_URL to the GLM service root (no /v1)}"
: "${GLM_API_KEY:?set GLM_API_KEY to the GLM service bearer token}"
BASE="${GLM_BASE_URL%/}"
BASE="${BASE%/v1}"

preflight() {
  # Fails if the token is wrong or the model id is not served. Prints no URL or token.
  local models
  models="$(curl -sf --max-time 20 -H "Authorization: Bearer ${GLM_API_KEY}" "${BASE}/v1/models")" \
    || { echo "claude-glm: /v1/models request failed (token or service down)." >&2; return 1; }
  grep -q "\"${MODEL}\"" <<<"$models" \
    || { echo "claude-glm: model ${MODEL} is not served by this endpoint." >&2; return 1; }
  echo "claude-glm: ${MODEL} is up." >&2
}

fresh_demo_copy() {
  # A new git repo each run, so the agent sees a clean tree and `git diff` shows only its fix.
  local dir
  dir="$(mktemp -d "${TMPDIR:-/tmp}/kv-prefix-cache.XXXX")"
  cp "${HERE}"/demo-bug/*.py "$dir/"
  git -C "$dir" init -q
  git -C "$dir" add -A
  git -C "$dir" -c user.name=demo -c user.email=demo@example.com commit -qm "Prefix cache index"
  echo "$dir"
}

preflight
[[ "${1:-}" == "--check" ]] && exit 0
command -v claude >/dev/null 2>&1 || { echo "claude-glm: claude CLI not on PATH." >&2; exit 1; }

WORKDIR="${1:-$(fresh_demo_copy)}"

export CLAUDE_CONFIG_DIR="${WEBINAR_CLAUDE_CONFIG_DIR:-$HOME/.claude-webinar}"
export ANTHROPIC_BASE_URL="$BASE"
export ANTHROPIC_AUTH_TOKEN="$GLM_API_KEY"
unset ANTHROPIC_API_KEY   # bearer token only; avoids the "both set" warning
export ANTHROPIC_MODEL="$MODEL"
# Map every tier to GLM so subagents and background tasks never fall back to Claude.
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_SUBAGENT_MODEL="$MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-600000}"   # 10 min: a long cold prefill must not time out

cd "$WORKDIR"
exec claude
