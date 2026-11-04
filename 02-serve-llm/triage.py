"""The cheap path: answer common questions from a lookup table, without a GPU.

Nothing here knows an LLM exists. It is an ordinary CPU deployment, and
composition.py binds it next to the LLM through the same `.bind()` seam.
"""

from ray import serve

# Questions the cheap path settles without reaching the GPU, keyed by a trigger word.
FAQ = {
    "hours": "Support is staffed 09:00 to 17:00 UTC, Monday through Friday.",
    "pricing": "Pricing is usage based and billed per GPU hour.",
    "status": "Live service status is published on the status page.",
}


def match_faq(text: str) -> str | None:
    """Return the FAQ answer this text can be settled with, or None to escalate."""
    lowered = text.lower()
    return next((answer for key, answer in FAQ.items() if key in lowered), None)


@serve.deployment(ray_actor_options={"num_cpus": 1})
class Triage:
    """CPU only, so it scales independently of the GPU deployment."""

    async def answer(self, text: str) -> str | None:
        return match_faq(text)
