#!/usr/bin/env python3
"""Prepare a cloned Olist course repo on an Azure ML compute instance."""

import argparse
import json
import shutil
import stat
import subprocess
import sys
import tempfile
import venv
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_PARENT = ROOT / "datasets" / "olist"
DATA_DIR = DATA_PARENT / "Ecommerce Dataset"
ARCHIVE = DATA_PARENT / "Ecommerce Dataset.zip"
ASSET_YAML = DATA_PARENT / "olist-data-asset.yml"
REQUIREMENTS = Path(__file__).with_name("requirements.txt")
COMPUTE_YAML = Path(__file__).with_name("compute.yml")
CSV_NAMES = frozenset(
    {
        "olist_customers_dataset.csv",
        "olist_geolocation_dataset.csv",
        "olist_order_items_dataset.csv",
        "olist_order_payments_dataset.csv",
        "olist_order_reviews_dataset.csv",
        "olist_orders_dataset.csv",
        "olist_products_dataset.csv",
        "olist_sellers_dataset.csv",
        "product_category_name_translation.csv",
    }
)
EXPECTED_MEMBERS = {f"Ecommerce Dataset/{name}" for name in CSV_NAMES}


class BootstrapError(Exception):
    """An actionable setup error without subprocess output or credentials."""


def verify_data(directory):
    if directory.is_symlink() or not directory.is_dir():
        raise BootstrapError(f"Expected a regular directory at {directory}.")
    found = {file.name for file in directory.glob("*.csv")}
    if found != CSV_NAMES:
        raise BootstrapError("The raw Olist folder must contain exactly the nine original CSVs; move generated files elsewhere.")
    for name in CSV_NAMES:
        file = directory / name
        if file.is_symlink() or not file.is_file() or file.stat().st_size == 0:
            raise BootstrapError(
                f"Missing, empty, or unsafe Olist CSV: {file}. "
                "Move the incomplete extracted directory aside and rerun bootstrap."
            )


def extract_data():
    if DATA_DIR.exists() or DATA_DIR.is_symlink():
        verify_data(DATA_DIR)
        print(f"Verified all nine Olist CSVs in {DATA_DIR}")
        return
    if not ARCHIVE.is_file():
        raise BootstrapError(f"Missing {ARCHIVE}. Copy the original Olist ZIP into datasets/olist/.")

    try:
        with zipfile.ZipFile(ARCHIVE) as source:
            members = {}
            for info in source.infolist():
                name = info.filename
                kind = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir():
                    if name != "Ecommerce Dataset/" or kind not in (0, stat.S_IFDIR):
                        raise BootstrapError("Unexpected directory in Olist ZIP.")
                    continue
                if (
                    name not in EXPECTED_MEMBERS
                    or name in members
                    or kind not in (0, stat.S_IFREG)
                ):
                    raise BootstrapError(
                        "Unsafe or unexpected ZIP entry; expected only the nine Olist CSVs "
                        "under 'Ecommerce Dataset/'."
                    )
                members[name] = info
            if set(members) != EXPECTED_MEMBERS:
                raise BootstrapError("Olist ZIP must contain exactly the nine original CSVs.")

            with tempfile.TemporaryDirectory(prefix=".olist-extract-", dir=DATA_PARENT) as tmp:
                extracted = Path(tmp) / "Ecommerce Dataset"
                extracted.mkdir()
                for name, info in members.items():
                    with source.open(info) as input_file, (extracted / Path(name).name).open("wb") as output_file:
                        shutil.copyfileobj(input_file, output_file)
                verify_data(extracted)
                if DATA_DIR.exists() or DATA_DIR.is_symlink():
                    raise BootstrapError(f"Destination appeared during extraction: {DATA_DIR}. Rerun bootstrap.")
                extracted.rename(DATA_DIR)
        print(f"Extracted and verified all nine Olist CSVs in {DATA_DIR}")
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise BootstrapError("Could not extract the Olist ZIP; check that it is intact and readable.") from exc


def run(command, error):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise BootstrapError(error) from exc
    if result.returncode != 0:
        raise BootstrapError(f"{error} (exit code {result.returncode}).")
    return result.stdout


def install_environment():
    if sys.version_info[:2] not in ((3, 10), (3, 11)):
        raise BootstrapError("Package installation requires Python 3.10 or 3.11. Use python3.10 or python3.11.")
    environment = ROOT / ".venv"
    python = environment / "bin" / "python"
    if not python.is_file():
        if environment.exists():
            raise BootstrapError(f"Existing {environment} has no bin/python. Fix or move it before rerunning.")
        print(f"Creating {environment}")
        try:
            venv.create(environment, with_pip=True)
        except subprocess.CalledProcessError as exc:
            raise BootstrapError("Could not create .venv with pip; install the matching python3-venv package.") from exc
    version = run(
        [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        "Cannot run the repository virtual environment",
    ).strip()
    if version not in ("3.10", "3.11"):
        raise BootstrapError("Existing .venv must use Python 3.10 or 3.11. Move it before rerunning.")
    print("Installing Python dependencies into .venv (this may take a few minutes)")
    run([str(python), "-m", "pip", "install", "-r", str(REQUIREMENTS)], "Dependency installation failed")
    run(
        [
            str(python), "-m", "ipykernel", "install", "--user",
            "--name", "olist-azure-ml", "--display-name", "Python (Olist Azure ML)",
        ],
        "Kernel registration failed",
    )
    print("Registered Studio kernel: Python (Olist Azure ML)")


def write_config(args):
    config = {
        "subscription_id": args.subscription,
        "resource_group": args.resource_group,
        "workspace_name": args.workspace,
    }
    (ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Wrote ignored config.json for MLClient.from_config()")


def az(args, *parts):
    return [
        "az", "ml", *parts, "--subscription", args.subscription,
        "--resource-group", args.resource_group, "--workspace-name", args.workspace,
        "--output", "json",
    ]


def show_or_missing(command, what):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise BootstrapError(f"Cannot check {what}; install Azure CLI and the ml extension.") from exc
    if result.returncode == 0:
        return True
    # Fail closed for permissions and service errors rather than risking an update.
    message = (result.stderr + result.stdout).lower()
    if any(marker in message for marker in ("notfound", "not found", "does not exist", "no results found")):
        return False
    raise BootstrapError(f"Cannot determine whether {what} exists; it was not created. Check Azure permissions.")


def configure_azure(args):
    if not shutil.which("az"):
        raise BootstrapError("Azure CLI is not installed or not on PATH. Install it before cloud setup.")
    run(["az", "account", "show", "--subscription", args.subscription, "--output", "json"],
        "Azure CLI is not logged in to this subscription; run az login")
    run(["az", "extension", "show", "--name", "ml", "--output", "json"],
        "Azure CLI ml extension is missing; run az extension add --name ml")
    run([
        "az", "ml", "workspace", "show", "--name", args.workspace,
        "--subscription", args.subscription, "--resource-group", args.resource_group,
        "--output", "json",
    ],
        "Azure ML workspace not found or inaccessible; check subscription, group, and name")
    print("Verified Azure CLI login, ml extension, and workspace")

    if not ASSET_YAML.is_file():
        raise BootstrapError(f"Missing {ASSET_YAML}; add the course asset definition before Azure registration.")
    asset = "olist-ecommerce-raw:1"
    if show_or_missing(az(args, "data", "show", "--name", "olist-ecommerce-raw", "--version", "1"), asset):
        print(f"Data asset {asset} already exists; leaving it unchanged")
    else:
        run(
            az(
                args, "data", "create", "--file", str(ASSET_YAML),
                "--name", "olist-ecommerce-raw", "--version", "1",
                "--type", "uri_folder", "--path", str(DATA_DIR),
            ),
            f"Could not register {asset}; verify the asset YAML and workspace permissions",
        )
        print(f"Registered data asset {asset}")

    if args.create_cluster:
        cluster = "olist-rai-cpu"
        if show_or_missing(az(args, "compute", "show", "--name", cluster), cluster):
            print(f"Compute {cluster} already exists; leaving it unchanged")
        else:
            run(az(args, "compute", "create", "--file", str(COMPUTE_YAML)),
                f"Could not create {cluster}; check regional CPU quota")
            print(f"Created scale-to-zero compute {cluster}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subscription", help="Azure subscription ID")
    parser.add_argument("--resource-group", help="Azure resource group containing the workspace")
    parser.add_argument("--workspace", help="Existing Azure ML workspace name")
    parser.add_argument("--skip-install", action="store_true", help="Extract only; do not create .venv, install packages, or register a kernel")
    parser.add_argument("--skip-azure", action="store_true", help="Do not call Azure CLI or create cloud resources")
    parser.add_argument("--create-cluster", action="store_true", help="Opt in to creating the scale-to-zero olist-rai-cpu cluster")
    args = parser.parse_args()
    settings = (args.subscription, args.resource_group, args.workspace)
    if any(settings) and not all(settings):
        parser.error("provide --subscription, --resource-group, and --workspace together")
    if not args.skip_azure and not all(settings):
        parser.error("Azure setup requires --subscription, --resource-group, and --workspace (or use --skip-azure)")
    if args.skip_azure and args.create_cluster:
        parser.error("--create-cluster cannot be combined with --skip-azure")

    try:
        extract_data()
        if not args.skip_install:
            install_environment()
        if all(settings):
            write_config(args)
        if not args.skip_azure:
            configure_azure(args)
    except BootstrapError as exc:
        parser.exit(1, f"Bootstrap error: {exc}\n")
    except OSError as exc:
        parser.exit(1, f"Bootstrap error: Local filesystem operation failed: {exc.strerror or 'check permissions and free space'}\n")
    print("Bootstrap complete")


if __name__ == "__main__":
    main()
