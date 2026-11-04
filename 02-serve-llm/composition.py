"""An LLM as one deployment inside a larger Python app.

`build_llm_deployment` returns the LLM as an ordinary deployment, so it binds
like any other. A CPU triage step answers what it can, and a plain Python `if`
sends the rest to the model. Two endpoints:
  - POST /ask         one JSON body with the whole answer.
  - POST /ask/stream  OpenAI-style server-sent events, forwarded as tokens arrive.

Run in a workspace:
    serve run composition:app
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"text": "what are your support hours?"}'
"""

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ray import serve
from ray.serve.handle import DeploymentHandle
from ray.serve.llm import LLMConfig, build_llm_deployment
from ray.serve.llm.openai_api_models import ChatCompletionRequest

from engine import ENGINE_KWARGS, MODEL_ID, MODEL_SOURCE
from triage import Triage

DONE = "data: [DONE]\n\n"

api = FastAPI()


class Ask(BaseModel):
    text: str


def sse_chunk(text: str) -> str:
    """One chat.completion.chunk event, so a triage answer streams like a model answer."""
    chunk = {
        "object": "chat.completion.chunk",
        "model": MODEL_ID,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


def chat_request(text: str, stream: bool) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model=MODEL_ID,
        messages=[{"role": "user", "content": text}],
        max_tokens=128,
        stream=stream,
        # A support answer needs no reasoning trace; skip it for latency.
        chat_template_kwargs={"enable_thinking": False},
    )


@serve.deployment
@serve.ingress(api)
class Assistant:
    """The ingress. Both collaborators arrive as plain DeploymentHandles."""

    def __init__(self, triage: DeploymentHandle, llm: DeploymentHandle) -> None:
        self.triage = triage
        # LLMServer.chat is an async generator, so the handle must stream.
        self.llm = llm.options(stream=True)

    @api.post("/ask")
    async def ask(self, body: Ask) -> dict:
        answer = await self.triage.answer.remote(body.text)
        if answer is not None:
            return {"answer": answer, "used_llm": False}
        # With stream=False the engine yields exactly one ChatCompletionResponse.
        async for response in self.llm.chat.remote(chat_request(body.text, stream=False)):
            return {"answer": response.choices[0].message.content, "used_llm": True}

    @api.post("/ask/stream")
    async def ask_stream(self, body: Ask) -> StreamingResponse:
        answer = await self.triage.answer.remote(body.text)
        if answer is not None:
            return StreamingResponse(iter([sse_chunk(answer), DONE]), media_type="text/event-stream")

        async def events():
            # With stream=True the engine yields lists of ready-made "data: ..." strings.
            async for batch in self.llm.chat.remote(chat_request(body.text, stream=True)):
                yield "".join(batch)
            yield DONE

        return StreamingResponse(events(), media_type="text/event-stream")


llm_config = LLMConfig(
    model_loading_config=dict(model_id=MODEL_ID, model_source=MODEL_SOURCE),
    deployment_config=dict(autoscaling_config=dict(min_replicas=1, max_replicas=1)),
    engine_kwargs=ENGINE_KWARGS,
)

app = Assistant.bind(Triage.bind(), build_llm_deployment(llm_config))
