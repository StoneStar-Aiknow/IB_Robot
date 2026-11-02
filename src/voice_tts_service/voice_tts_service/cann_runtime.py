"""Resolve the CANN runtime a deployment's Ascend artifacts were built against.

An OM is compiled ahead of time against one CANN ABI and the manifest records it as
``target.runtime_abi`` (e.g. ``cann-8.1.RC1``). A host may have several toolkits installed, and
``/usr/local/Ascend/ascend-toolkit/latest`` is not necessarily the one an artifact needs: loading an
OM under a different runtime can fail inside ACL, including by crashing the process rather than
returning an error. Node environments are therefore derived from the declared ABI, never from
whatever the caller happened to source.

This selects an already-installed toolkit; it never modifies the shared installation.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from robot_config.logger_utils import get_colored_logger

logger = get_colored_logger("voice_tts_service.cann_runtime")

#: Where CANN toolkits are installed, one directory per version plus the ``latest`` selection.
#: Overridable so a host that installs elsewhere does not need a code change.
CANN_INSTALL_ROOT = Path(os.environ.get("IBROBOT_CANN_INSTALL_ROOT", "/usr/local/Ascend/ascend-toolkit"))


def cann_version_from_abi(runtime_abi: str | None) -> str | None:
    """Return the toolkit version named by a ``cann-<version>`` ABI string."""

    if not runtime_abi:
        return None
    match = re.fullmatch(r"cann-(?P<version>[0-9][0-9A-Za-z.]*)", runtime_abi.strip())
    return match.group("version") if match else None


def resolve_cann_root(runtime_abi: str | None, *, install_root: Path | None = None) -> Path | None:
    """Pick the toolkit for ``runtime_abi``; return ``None`` when no usable one is installed.

    A missing toolkit is reported rather than silently substituted. Loading an OM under a CANN
    other than the one it was built against can abort the process inside ACL instead of raising,
    so pointing the node at a different (or nonexistent) tree would hide the cause of the crash.
    Returning ``None`` leaves the node with the caller's environment and logs why.
    """

    root = install_root if install_root is not None else CANN_INSTALL_ROOT
    version = cann_version_from_abi(runtime_abi)
    if version:
        candidate = root / version
        if candidate.is_dir():
            logger.info(f"CANN {version} selected from the declared ABI {runtime_abi!r}: {candidate}")
            return candidate
        logger.warning(
            f"model declares ABI {runtime_abi!r} but {candidate} is not installed; "
            "leaving the CANN environment to the caller, which may not match the compiled artifacts"
        )
        return None
    if runtime_abi:
        logger.warning(f"unrecognized CANN ABI {runtime_abi!r}; leaving the CANN environment to the caller")
    else:
        logger.warning("Ascend deployment declares no target.runtime_abi; leaving the CANN environment to the caller")
    return None


def _is_within(path: Path, root: Path) -> bool:
    """Whether ``path`` names ``root`` or something inside it, after resolving symlinks."""

    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def ascend_node_environment(cann_root: Path | None) -> dict[str, str]:
    """Environment that makes a node use ``cann_root`` regardless of the caller's shell.

    ``pyACL`` ships its ``acl`` module outside ``python/site-packages`` in some toolkits, so both
    locations are exposed; only existing paths are added. A ``None`` root (no usable toolkit, see
    :func:`resolve_cann_root`) contributes nothing rather than naming a directory that is absent.

    The caller's own entries are kept behind the selected toolkit's, except the ones that belong to
    another installed toolkit: a shell that sourced ``.../ascend-toolkit/latest`` (or any other
    version) would otherwise leave that tree in ``PYTHONPATH``/``LD_LIBRARY_PATH``/``PATH`` for the
    rest of the process. Resolution is ordered, so the selected root wins for anything it ships, but
    a library or module that only the other version has would still be loaded from it, mixing two
    CANN ABIs in one process.
    """

    if cann_root is None or not cann_root.is_dir():
        return {}

    # One directory per version, so the install root holding every toolkit is the selected root's
    # parent (see :data:`CANN_INSTALL_ROOT`).
    install_root = cann_root.parent

    def prepend(name: str, entries: tuple[Path, ...]) -> str:
        values = [str(path) for path in entries if path.exists()]
        dropped = []
        for entry in (os.environ.get(name) or "").split(os.pathsep):
            if not entry or entry in values:
                continue
            if _is_within(Path(entry), install_root) and not _is_within(Path(entry), cann_root):
                dropped.append(entry)
                continue
            values.append(entry)
        if dropped:
            logger.warning(
                f"{name}: ignoring {len(dropped)} inherited path(s) from another installed CANN toolkit: "
                f"{', '.join(dropped)}"
            )
        return os.pathsep.join(values)

    environment = {
        "ASCEND_TOOLKIT_HOME": str(cann_root),
        "ASCEND_HOME_PATH": str(cann_root),
        "PYTHONPATH": prepend(
            "PYTHONPATH",
            (cann_root / "pyACL" / "python" / "site-packages", cann_root / "python" / "site-packages"),
        ),
        "LD_LIBRARY_PATH": prepend(
            "LD_LIBRARY_PATH",
            (cann_root / "lib64", cann_root / "runtime" / "lib64", cann_root / "aarch64-linux" / "lib64"),
        ),
        "PATH": prepend("PATH", (cann_root / "bin", cann_root / "aarch64-linux" / "bin")),
    }
    for name, path in (("ASCEND_OPP_PATH", cann_root / "opp"), ("ASCEND_AICPU_PATH", cann_root)):
        if path.exists():
            environment[name] = str(path)
    return environment


__all__ = ["CANN_INSTALL_ROOT", "ascend_node_environment", "cann_version_from_abi", "resolve_cann_root"]
