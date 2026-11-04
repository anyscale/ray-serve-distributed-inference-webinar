#!/usr/bin/env bash
# Claude Code against the service's /v1/messages (Anthropic Messages API).
# Adapted from anyscale/llm-serving-for-coding-agents part2-connect-clients-production/claude-service.sh.
#
#   export ANYSCALE_BASE_URL=https://<service-host>/v1   # console: Services -> service -> Query
#   export ANYSCALE_API_KEY=<service bearer token>
#   ./claude-code.sh                                     # extra args pass through to claude
set -euo pipefail
: "${ANYSCALE_BASE_URL:?set ANYSCALE_BASE_URL to the service URL ending in /v1}"
: "${ANYSCALE_API_KEY:?set ANYSCALE_API_KEY to the service bearer token}"
MODEL="${ANYSCALE_MODEL:-qwen3.6-27b}"

# Claude Code appends /v1/messages itself, so it wants the service root.
export ANTHROPIC_BASE_URL="${ANYSCALE_BASE_URL%/v1}"
export ANTHROPIC_AUTH_TOKEN="$ANYSCALE_API_KEY"
unset ANTHROPIC_API_KEY
# Map every model tier to the one served model, so subagents land there too.
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
# Claude Code doesn't know qwen3.6-27b's window: it would assume 200K and warn at startup.
# 262144 matches service-agents.yaml; use 131072 against the L40S fallback.
export CLAUDE_CODE_MAX_CONTEXT_TOKENS="${CLAUDE_CODE_MAX_CONTEXT_TOKENS:-262144}"
# 20 minutes, so a first request during a cold start does not time out.
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-1200000}"

exec claude "$@"
