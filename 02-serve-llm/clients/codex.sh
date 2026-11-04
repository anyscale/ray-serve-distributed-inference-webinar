#!/usr/bin/env bash
# Codex against the service's /v1/responses (OpenAI Responses API).
# Adapted from anyscale/llm-serving-for-coding-agents part2-connect-clients-production/codex-service.sh.
#
#   export ANYSCALE_BASE_URL=https://<service-host>/v1   # console: Services -> service -> Query
#   export ANYSCALE_API_KEY=<service bearer token>
#   ./codex.sh                                           # extra args pass through to codex
# Set CODEX_MODEL_CONTEXT_WINDOW=131072 against the L40S fallback service.
set -euo pipefail
: "${ANYSCALE_BASE_URL:?set ANYSCALE_BASE_URL to the service URL ending in /v1}"
: "${ANYSCALE_API_KEY:?set ANYSCALE_API_KEY to the service bearer token}"
export ANYSCALE_API_KEY   # codex reads the token from the env var named by env_key
MODEL="${ANYSCALE_MODEL:-qwen3.6-27b}"
PROVIDER=anyscale

# Hosted OpenAI tools call routes this service does not serve, so they are off.
exec codex \
  -c model="$MODEL" \
  -c model_provider="$PROVIDER" \
  -c "model_providers.$PROVIDER.name=Anyscale" \
  -c "model_providers.$PROVIDER.base_url=${ANYSCALE_BASE_URL%/}" \
  -c "model_providers.$PROVIDER.env_key=ANYSCALE_API_KEY" \
  -c "model_providers.$PROVIDER.wire_api=responses" \
  -c "model_providers.$PROVIDER.requires_openai_auth=false" \
  -c model_context_window="${CODEX_MODEL_CONTEXT_WINDOW:-262144}" \
  -c tools.web_search=false \
  -c features.image_generation=false \
  "$@"
