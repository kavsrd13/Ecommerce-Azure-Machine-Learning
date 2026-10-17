#!/usr/bin/env python3
"""Submit one explicitly selected Olist Azure ML job from the cloned course repo."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
JOBS = {
    "freight": ROOT / "exercises/03-freight-training/jobs/train-job.yml",
    "delivery": ROOT / "exercises/04-delivery-time-pipeline/pipelines/delivery-time-pipeline.yml",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exercise", choices=JOBS, help="Only this exercise's job will be submitted")
    parser.add_argument("--dry-run", action="store_true", help="Show the command without using Azure")
    args = parser.parse_args()
    config_path = ROOT / "config.json"
    if not config_path.is_file():
        parser.error("Run scripts/bootstrap.py with your workspace arguments first")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    executable = shutil.which("az")
    if executable is None:
        parser.error("Install Azure CLI and the ml extension on this machine first")
    cli = [
        executable, "ml", "job", "create", "--file", str(JOBS[args.exercise]),
        "--subscription", config["subscription_id"],
        "--resource-group", config["resource_group"],
        "--workspace-name", config["workspace_name"],
        "--query", "{name:name,status:status,studio_url:studio_url}", "--output", "json",
    ]
    if args.dry_run:
        print(" ".join(cli))
        return
    result = subprocess.run(cli, check=True, capture_output=True, text=True)
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
