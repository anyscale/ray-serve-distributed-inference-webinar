"""Static checks for the production YAMLs in this directory.

Three levels, each run only where its imports are available:
  1. yaml.safe_load on every *.yaml (always).
  2. LLMConfig validation and an app build through the YAML's builder, without
     starting an engine (needs `ray.serve.llm`, which needs vLLM installed).
  3. engine_kwargs names against the installed vLLM's EngineArgs and FrontendArgs
     (Ray forwards server options such as tool_call_parser to the frontend). A
     miss is a warning: a config may target a newer vLLM than the one here.

    python check_configs.py            # exits 1 if a level-1 or level-2 check fails
"""

from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent


def load_all() -> dict[str, dict]:
    """Parse every YAML file here; a parse error is fatal."""
    docs = {}
    for path in sorted(HERE.glob("*.yaml")):
        docs[path.name] = yaml.safe_load(path.read_text())
        print(f"yaml ok      {path.name}")
    return docs


def llm_configs_of(args: dict) -> list[dict]:
    """The LLMConfig dicts in one application's builder args, for any builder."""
    if "llm_configs" in args:
        return args["llm_configs"]
    if "llm_config" in args:
        return [args["llm_config"]]
    return [args["prefill_config"], args["decode_config"]]


def resolve(import_path: str):
    """Import `module:attr` and return the attribute."""
    module, attr = import_path.split(":")
    return getattr(importlib.import_module(module), attr)


def check_builds(name: str, doc: dict) -> bool:
    """Validate each LLMConfig and build each application object."""
    from ray.serve.llm import LLMConfig

    ok = True
    for app in doc.get("applications", []):
        try:
            for cfg in llm_configs_of(app["args"]):
                LLMConfig.model_validate(cfg)
            resolve(app["import_path"])(app["args"])
            print(f"build ok     {name} [{app['name']}]")
        except Exception as exc:  # noqa: BLE001 - report every failure
            print(f"build FAIL   {name} [{app['name']}]: {type(exc).__name__}: {exc}"[:400])
            ok = False
    return ok


def check_engine_kwargs(name: str, doc: dict, known: set[str]) -> None:
    """Warn about engine_kwargs keys the installed vLLM does not define."""
    for app in doc.get("applications", []):
        for cfg in llm_configs_of(app["args"]):
            unknown = sorted(set(cfg.get("engine_kwargs", {})) - known)
            label = f"{name} [{app['name']}]"
            print(f"vllm {'WARN' if unknown else 'ok  '}    {label}"
                  + (f": not in this vLLM: {unknown}" if unknown else ""))


def vllm_arg_names() -> set[str]:
    """Field names of vLLM's EngineArgs plus its OpenAI server's FrontendArgs."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.openai.cli_args import FrontendArgs

    return {f.name for cls in (EngineArgs, FrontendArgs) for f in dataclasses.fields(cls)}


def available(module: str) -> bool:
    """True if `module` imports; prints why not otherwise."""
    try:
        importlib.import_module(module)
        return True
    except Exception as exc:  # noqa: BLE001 - any import failure means skip
        print(f"skip         {module}: {type(exc).__name__}: {exc}"[:200])
        return False


def main() -> int:
    """Run the levels whose imports are available; return the exit code."""
    docs = load_all()
    if not available("ray.serve.llm"):
        return 0
    ok = all([check_builds(name, doc) for name, doc in docs.items()])
    if available("vllm"):
        from vllm import __version__ as vllm_version

        print(f"vLLM {vllm_version}")
        known = vllm_arg_names()
        for name, doc in docs.items():
            check_engine_kwargs(name, doc, known)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
