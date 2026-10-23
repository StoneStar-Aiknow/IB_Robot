"""Install only missing selected-backend system packages during explicit set."""

import os
import re
import shutil
import subprocess
import sys

from .platform_facts import FactsError, package_status


def _run(command, *, timeout=60):
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env={**os.environ, "LC_ALL": "C"})
    return result


def _installation_plan(specs, debian):
    if debian:
        command = ["apt-get", "-s", "--no-install-recommends", "--no-remove", "--no-upgrade", "install", *specs]
    else:
        command = ["dnf", "--assumeno", "install", *specs]
    result = _run(command, timeout=120)
    output = result.stdout + "\n" + result.stderr
    if debian:
        planned = re.findall(r"^Inst (\S+)", output, re.MULTILINE)
        if result.returncode or not planned or re.search(r"^Remv ", output, re.MULTILINE):
            raise FactsError(f"Unsafe apt install plan: {output.strip()}")
        for package in planned:
            existing = _run(["dpkg-query", "-W", "-f=${db:Status-Status}", package], timeout=5)
            if existing.stdout == "installed":
                raise FactsError(f"Install plan would change installed package: {package}")
            if existing.returncode and "no packages found matching" not in existing.stderr:
                raise FactsError(f"Cannot verify install plan for {package}: {existing.stderr.strip()}")
    else:
        # DNF's --assumeno exits with 1 for an otherwise valid transaction.
        # Allow installs only; all other transaction verbs fail closed.
        if (
            "Transaction Summary" not in output
            or not re.search(r"^Install\s+\d+ Packages?\s*$", output, re.MULTILINE)
            or re.search(r"^(Upgrade|Downgrade|Remove|Reinstall|Obsolet)\s+\d+ Packages?\s*$", output, re.MULTILINE)
            or re.search(r"^(Upgrading|Downgrading|Removing|Reinstalling|Obsoleting):", output, re.MULTILINE)
            or result.returncode not in (0, 1)
            or (result.returncode == 1 and "Operation aborted" not in output)
        ):
            raise FactsError(f"Unsafe or unrecognized dnf install plan: {output.strip()}")
    return output.strip()


def install_if_missing(backend):
    """The current set may install packages; caller rechecks runtime before applying."""
    _, missing = package_status(backend)
    if not missing:
        return False
    debian = shutil.which("dpkg-query") is not None
    specs = missing
    manager = "apt" if debian else "dnf"
    print(
        f"DDS：检测到缺少 {', '.join(specs)}；正在检查 {manager} 仓库并计算安装计划（可能需要联网，请稍候）...",
        file=sys.stderr,
        flush=True,
    )
    plan = _installation_plan(specs, debian)
    try:
        with open("/dev/tty", encoding="utf-8") as reader, open("/dev/tty", "w", encoding="utf-8") as writer:
            print(f"\nDDS {backend} 缺少依赖；将安装：{', '.join(specs)}", file=writer)
            print(f"包管理器计划：\n{plan}\n系统包安装不会随 DDS unset 而卸载。", file=writer)
            print("现在安装？Y/n: ", end="", flush=True, file=writer)
            line = reader.readline()
            if not line:
                raise FactsError("No installation confirmation; DDS configuration unchanged")
            answer = line.strip().lower()
    except OSError as exc:
        raise FactsError("Missing DDS packages require a terminal to confirm installation") from exc
    if answer not in ("", "y", "yes"):
        raise FactsError("DDS dependency installation cancelled; configuration unchanged")
    # Recheck the plan immediately before sudo; do not reinstall or change an
    # existing package. No separate rollback system is added for OS packages.
    if package_status(backend)[1] != missing:
        raise FactsError("Package state or install plan changed; retry set")
    print(f"DDS：正在复核安装计划（{manager}，可能需要联网，请稍候）...", file=sys.stderr, flush=True)
    _installation_plan(specs, debian)
    elevation = [] if os.geteuid() == 0 else ["sudo"]
    command = (
        [*elevation, "apt-get", "-y", "--no-install-recommends", "--no-remove", "--no-upgrade", "install", *specs]
        if debian
        else [*elevation, "dnf", "-y", "install", *specs]
    )
    print(f"DDS：开始安装 {', '.join(specs)}（{manager}）...", file=sys.stderr, flush=True)
    # shell_entry.py runs inside Bash command substitution; never emit package
    # manager progress to its stdout (reserved for the shell's action token).
    result = subprocess.run(command, stdout=sys.stderr, stderr=sys.stderr)
    if result.returncode:
        raise FactsError(
            "DDS package installation failed; check the package manager output; no DDS settings were applied"
        )
    if package_status(backend)[1]:
        raise FactsError(
            "DDS package installation finished but required packages are still missing; DDS settings unchanged"
        )
    return True
