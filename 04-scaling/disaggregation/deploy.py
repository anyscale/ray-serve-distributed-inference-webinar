"""Deploy one builder from app.py, wait until it is RUNNING, then exit.

The Serve app outlives this script, so a shell can deploy, measure and tear
down in sequence without a background `serve run`:

    python deploy.py collocated     # or: python deploy.py disaggregated
    python itl.py --label collocated
    serve shutdown -y
"""

import argparse

import ray
from ray import serve

from app import PDArgs, build_app, build_collocated_app

BUILDERS = {"collocated": build_collocated_app, "disaggregated": build_app}


def main(which: str) -> None:
    """Deploy the chosen app at route / and print its status."""
    ray.init(log_to_driver=False)  # replica logs stay in their files
    serve.run(BUILDERS[which](PDArgs()), name=which, route_prefix="/",
              logging_config={"log_level": "WARNING"}, blocking=False)
    print(f"{which}: {serve.status().applications[which].status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("which", choices=sorted(BUILDERS))
    main(parser.parse_args().which)
