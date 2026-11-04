"""Offline batch inference with ray.data.llm, using the service's engine settings.

Ray Data streams a dataset through a pool of vLLM engine actors. The engine
arguments are imported from engine.py, the same module the online services
use, so online and offline share settings, not objects.

Runs as an Anyscale Job (see job-batch.yaml), or in a GPU workspace:
    python batch_infer.py
"""

import ray
from ray.data.llm import build_processor, vLLMEngineProcessorConfig

from engine import ENGINE_KWARGS, MODEL_SOURCE

QUESTIONS = [
    "Summarize what Ray Serve does in one sentence.",
    "What is a KV cache?",
    "Name two reasons to separate prefill from decode.",
    "What does tensor parallelism split?",
    "Why do coding agents benefit from prefix caching?",
    "What is continuous batching?",
]


def build_llm_processor():
    config = vLLMEngineProcessorConfig(
        model_source=MODEL_SOURCE,
        engine_kwargs=ENGINE_KWARGS,
        # One engine actor; pass a (min, max) tuple to autoscale over a large dataset.
        concurrency=1,
        batch_size=32,
        # Same switch as the composition app: no reasoning trace for short answers.
        chat_template_stage={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return build_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": row["question"]},
            ],
            sampling_params=dict(temperature=0.0, max_tokens=96),
        ),
        # The engine stage writes the templated text to a "prompt" column, so the
        # input column is named "question" to survive to the output.
        postprocess=lambda row: dict(question=row["question"], answer=row["generated_text"]),
    )


def main() -> None:
    ds = ray.data.from_items([{"question": q} for q in QUESTIONS])
    for row in build_llm_processor()(ds).take_all():
        print(f"Q: {row['question']}\nA: {row['answer']}\n")


if __name__ == "__main__":
    main()
