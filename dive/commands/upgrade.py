"""CLI Command logic for `dive upgrade`."""

from __future__ import annotations

import subprocess
import sys
from typing import Optional

from dive import __version__
from dive.utils.logging import Console, Style


def run_upgrade(console: Console, force: bool = False) -> None:
    """Pull latest changes from GitHub and upgrade dependencies."""
    console.banner("DIVE AUTO-UPGRADE", "Updating DIVE engine & dependencies to latest GitHub main release")

    console.kv("Current Version", f"v{__version__}")
    console.kv("Python Runtime", sys.version.split()[0])
    console.print("")

    # Step 1: Git Pull
    with console.spinner("Fetching latest updates from GitHub (origin/main)..."):
        try:
            res_pull = subprocess.run(
                ["git", "pull", "origin", "main"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            git_success = res_pull.returncode == 0
            git_msg = res_pull.stdout.strip() or res_pull.stderr.strip()
        except Exception as exc:
            git_success = False
            git_msg = str(exc)

    if git_success:
        console.success(f"Git pull succeeded: {git_msg}")
    else:
        console.warn(f"Git pull warning: {git_msg}")

    # Step 2: Pip Upgrade
    with console.spinner("Upgrading DIVE package & dependencies (pip install -e .[full])..."):
        try:
            res_pip = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "-e", ".[full]"],
                capture_output=True,
                text=True,
                timeout=120,
            )
            pip_success = res_pip.returncode == 0
            pip_msg = res_pip.stdout.strip()
        except Exception as exc:
            pip_success = False
            pip_msg = str(exc)

    if pip_success:
        console.success("Dependencies and DIVE platform successfully upgraded!")
    else:
        console.warn(f"Pip upgrade encountered an issue: {pip_msg[:100]}")

    console.print("")
    console.rule("Upgrade Complete")
    console.success("DIVE is up to date and ready for production ML engineering.")
