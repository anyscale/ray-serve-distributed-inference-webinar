"""Build each service.yaml's application without starting an engine.

For kv-aware-routing/ and pd-disaggregation/: parse service.yaml, validate its
`args:` against the builder's argument model, and call the builder. Needs
`ray.serve.llm`, which needs vLLM installed, so run it on the ray-llm image:

    cd 04-scaling && python checks/check_apps.py    # exits 1 on any failure
"""

import importlib
import inspect
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
APP_DIRS = ["kv-aware-routing", "pd-disaggregation"]


def check(app_dir: str) -> bool:
    """Build every application listed in one directory's service.yaml."""
    doc = yaml.safe_load((ROOT / app_dir / "service.yaml").read_text())
    sys.path.insert(0, str(ROOT / app_dir))
    sys.modules.pop("app", None)  # each directory has its own app.py
    ok = True
    for spec in doc["applications"]:
        module_name, attr = spec["import_path"].split(":")
        builder = getattr(importlib.import_module(module_name), attr)
        arg_model = next(iter(inspect.signature(builder).parameters.values())).annotation
        try:
            builder(arg_model(**spec.get("args", {})))
            print(f"build ok     {app_dir} [{spec['name']}] {spec['import_path']}")
        except Exception as exc:  # noqa: BLE001 - report every failure
            print(f"build FAIL   {app_dir} [{spec['name']}]: {type(exc).__name__}: {exc}"[:400])
            ok = False
    sys.path.pop(0)
    return ok


if __name__ == "__main__":
    sys.exit(0 if all([check(d) for d in APP_DIRS]) else 1)
