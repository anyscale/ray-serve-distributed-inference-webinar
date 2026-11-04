# 00 · Cold open: a coding agent that isn't Claude

Claude Code fixes a real bug in a small repo. Then the
reveal: every token came from GLM-5.3, an open-weight mixture-of-experts model
served by Ray Serve LLM on two 8× B200 nodes, through its Anthropic-compatible
`/v1/messages` endpoint.

Nothing is deployed here. It is configuration only: Claude Code is pointed at
an existing service with six environment variables.

## The configuration

Claude Code speaks the Anthropic Messages API. Ray Serve LLM with direct
streaming serves vLLM's own `/v1/messages` handler, so no translation proxy sits
in between. This is the whole change, as a Claude Code `settings.json` block:

```json
{
  "env": {
    "ANTHROPIC_BASE_URL": "https://<glm-service-root>",
    "ANTHROPIC_AUTH_TOKEN": "<service-bearer-token>",
    "ANTHROPIC_MODEL": "glm-5.3-nvfp4",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.3-nvfp4",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "glm-5.3-nvfp4",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "glm-5.3-nvfp4"
  }
}
```

- `ANTHROPIC_BASE_URL` is the service root, without `/v1`. Claude Code appends
  `/v1/messages` itself.
- The three `DEFAULT_*_MODEL` lines map every tier to GLM. Without them,
  subagents and background tasks would ask for a Claude model id the service
  does not serve.

## Running it

```bash
# Service root and token: Anyscale console -> Services -> your service -> Query.
export GLM_BASE_URL=...      # no trailing /v1
export GLM_API_KEY=...

./claude-glm.sh --check      # preflight: token valid and glm-5.3-nvfp4 served
./claude-glm.sh              # fresh copy of demo-bug/ in a temp git repo, then Claude Code
```

`claude-glm.sh` exports the variables above and starts Claude Code with a clean
config directory (`~/.claude-webinar`), so no personal `CLAUDE.md`, memory or
MCP servers load. Run it once beforehand to clear Claude Code's first-run
prompts in that directory.

The prompt used in the webinar:

> The prefix cache tests fail. Find the bug and fix it, then run the tests.

## The demo repo

`demo-bug/` is a 50-line prefix-cache index of the kind vLLM keeps for its KV
blocks, plus five unit tests (`python -m unittest -v`, standard library only).
One test fails. Spoiler: each block is hashed on its own tokens, but a block's
KV depends on the whole prefix before it, so the hash must chain the parent
block's hash. The fix is two lines.

The bug foreshadows the engine part of the talk: the agent fixes the very mechanism that
makes coding agents cheap to serve.

## If you demo this live

- Never show the service URL or token. Do not run `/status` in Claude Code, which
  prints the base URL. Show the model with `/model`.
- Use a large terminal font and a clean prompt.

## Source

The launcher is adapted from
`anyscale/llm-serving-for-coding-agents: part2-connect-clients-production/claude-service.sh`.
