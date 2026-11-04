"""What CANN release the host has, and whether a deployment's ABI matches it.

A deployment declares the ABI its artifact was compiled against
(``target.runtime_abi``, e.g. ``cann-8.1.RC1``), but nothing on the load path
used to compare that with the host. The mismatch then surfaced as an ACL
failure deep inside model loading -- measured on an Ascend 310P: an OM built by
CANN 8.3.RC1's ATC fails with error 500002 on an 8.1.RC1 host, after every
other node has already started. For a consumer whose absence is silent rather
than loud, that is the worst possible moment to find out.

It lives with the ACL backend rather than with the manifest schema: the
manifest is hardware-independent by definition, and what the host has
installed is exactly the hardware-dependent half. It complements the
``supported_runtime_abis`` admission the unified runtime already does, which
compares a deployment against what a backend *implementation* declares it can
run; this compares the same deployment against what this machine actually has.

The detection mirrors ``scripts/setup/detect.sh``'s ``detect_cann_version``,
including its file precedence, because setup and runtime disagreeing about the
host's release would be worse than either being wrong alone. Its fixtures
(``scripts/setup/tests/test_cann_torch_versions.sh``) are the same cases the
tests here cover.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

#: A public CANN release, such as ``8.1.RC1``. Internal build numbers
#: (``7.7.0.1.238``) carry four or more components and are not this.
_RELEASE = re.compile(r"^[0-9]+\.[0-9]+(\.[A-Za-z0-9]+)*$")

#: Where the toolkit root is looked for, in the order setup uses.
_ROOT_ENV = ("IBR_CANN_TOOLKIT_ROOT", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH")
_DEFAULT_ROOT = "/usr/local/Ascend/ascend-toolkit/latest"

#: Candidate version files under the toolkit root, in precedence order. The
#: merged ``latest`` directory is a tree of component symlinks rather than a
#: symlink to the release, so its public ``version.cfg`` comes first.
_VERSION_FILES = (
    "version.cfg",
    "ascend_toolkit_install.info",
    "aarch64-linux/ascend_toolkit_install.info",
    "arm64-linux/ascend_toolkit_install.info",
    "x86_64-linux/ascend_toolkit_install.info",
    "runtime/version.info",
    "toolkit/version.info",
)


def toolkit_root() -> Path:
    """The toolkit directory to inspect, honouring the same environment setup does."""
    for name in _ROOT_ENV:
        value = os.environ.get(name)
        if value:
            return Path(value)
    return Path(_DEFAULT_ROOT)


def _release_from_file(path: Path) -> str | None:
    """The public release a version file reports, or None when it reports none."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("version="):
            candidate = line[len("version=") :]
        elif line.startswith("version_dir="):
            candidate = line[len("version_dir=") :]
        elif "_running_version=[" in line and ":" in line:
            # runtime_running_version=[7.7.0.1.238:8.1.RC1] -- the public
            # release is after the colon; the internal build is not it.
            candidate = line.split(":", 1)[1].split("]", 1)[0]
        else:
            continue
        candidate = candidate.strip()
        if _RELEASE.match(candidate):
            return candidate
    return None


def detect_cann_release(root: str | Path | None = None) -> str | None:
    """The host's CANN release, or None when no toolkit can be identified.

    None is a real answer and not an error: a host with no NPU has no release,
    and a caller must decide for itself whether that is a reason to refuse.
    """
    base = Path(root) if root is not None else toolkit_root()
    for relative in _VERSION_FILES:
        release = _release_from_file(base / relative)
        if release is not None:
            return release
    # A versioned toolkit root names its own release; the merged "latest" does not.
    resolved = base.resolve() if base.exists() else base
    if _RELEASE.match(resolved.name):
        return resolved.name
    return None


def normalize_runtime_abi(value: str | None) -> str | None:
    """The release part of an ABI string: ``cann-8.1.RC1`` and ``8.1.RC1`` both give ``8.1.RC1``."""
    if not value:
        return None
    candidate = str(value).strip()
    for prefix in ("cann-", "cann_", "cann"):
        if candidate.lower().startswith(prefix):
            candidate = candidate[len(prefix) :].lstrip("-_")
            break
    return candidate or None


def _family(release: str) -> tuple[str, str] | None:
    parts = release.split(".")
    return (parts[0], parts[1]) if len(parts) >= 2 else None


def runtime_abi_matches(declared: str | None, host: str | None) -> bool:
    """Whether an artifact built for ``declared`` can be loaded on ``host``.

    Compared at major.minor, because that is the granularity at which the ACL
    failure was observed, and a patch release that did break compatibility
    would be a vendor regression rather than something to encode here. An
    unknown value on either side matches: refusing to load on a host whose
    release cannot be read would make the check worse than no check at all,
    and that decision belongs to the caller, which can see the None.
    """
    wanted = normalize_runtime_abi(declared)
    actual = normalize_runtime_abi(host)
    if wanted is None or actual is None:
        return True
    left, right = _family(wanted), _family(actual)
    if left is None or right is None:
        return wanted == actual
    return left == right


__all__ = [
    "detect_cann_release",
    "normalize_runtime_abi",
    "runtime_abi_matches",
    "toolkit_root",
]
