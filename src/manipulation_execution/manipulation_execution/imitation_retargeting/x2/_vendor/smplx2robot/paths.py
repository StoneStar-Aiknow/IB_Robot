"""Where the toolkit finds the things it cannot carry in the repo.

The ledger travels with the package (`data/x2_retarget_ledger.json`) because it
IS the deliverable of the retargeting stage. Three things do not:

* the X2 URDF/MJCF tree (`aimdk/...`) is a read-only vendor drop, and copying a
  robot model into a second repository is how two copies start disagreeing;
* the source animation is the user's own motion capture, not sample data;
* the ledger *archive* -- the original that `data/x2_retarget_ledger.json` was
  copied from -- lives with the retargeting write-up, and is only needed to
  check the copy against it.

All three are therefore *located* at run time. A path can be stated outright in
three ways, in this order of precedence:

    1. the command-line flag
    2. the environment variable
    3. `smplx2robot.json`, searched up from the working directory and then at
       ~/.config/smplx2robot/config.json

and if none of those says anything, an upward walk from the working directory
and from the package finds the usual layout with no configuration at all.

Nothing here ever guesses silently. A failed search raises with the list of
places that were tried, and a path that WAS stated outright but is not there is
an error rather than a reason to go looking for some other file -- falling back
would run the conversion against a different robot or a different clip than the
one that was asked for, and record that substitution nowhere the operator looks.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import pear

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR / "data"

LEDGER_PATH = DATA_DIR / "x2_retarget_ledger.json"
SCENE_TEMPLATE = DATA_DIR / "x2_scene_template.xml"

# Relative to the root of the vendor drop.
URDF_SUBPATH = Path("aimdk/extra/mc-rl/x2_urdf/X2_URDF-v1.3.0")
MODEL_FILE = "x2_ultra.xml"

ANIMATION_SUBPATH = Path("handoff_anim/animation/pear_robot_source_animation.npz")

# The archive copy of the ledger: the file the retargeting write-up maintains,
# and the one `data/x2_retarget_ledger.json` must agree with. See `verify.py`.
LEDGER_SOURCE_SUBPATH = Path("handoff_anim/wrist_axis_mapping.json")

ENV_URDF_DIR = "X2_URDF_DIR"
ENV_ANIMATION = "X2_SOURCE_ANIMATION"
ENV_LEDGER_SOURCE = "X2_LEDGER_SOURCE"
ENV_CONFIG = "SMPLX2ROBOT_CONFIG"

CONFIG_FILE = "smplx2robot.json"
CONFIG_KEYS: tuple[str, ...] = ("urdf_dir", "animation", "ledger_source")
USER_CONFIG = Path.home() / ".config" / "smplx2robot" / "config.json"

# How far up the tree to look. The vendor drop and the animation both live a
# couple of levels above a normal checkout; ten is far past that and still
# terminates on a deep path.
_MAX_ASCENT = 10


def _ancestors(start: Path) -> list[Path]:
    start = Path(start).resolve()
    chain = [start, *start.parents]
    return chain[:_MAX_ASCENT + 1]


def _search_roots() -> list[Path]:
    """Directories to walk up from: the working directory, then the package."""
    roots: list[Path] = []
    for start in (Path.cwd(), PACKAGE_DIR):
        for candidate in _ancestors(start):
            if candidate not in roots:
                roots.append(candidate)
    return roots


def config_path() -> Path | None:
    """The config file in effect, or None if there is none.

    Not cached: the working directory and the environment are both things a
    caller may legitimately change between calls, and a stale cache here would
    silently answer for the wrong tree.
    """
    chosen = os.environ.get(ENV_CONFIG)
    if chosen:
        path = Path(chosen).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"{ENV_CONFIG} is set to {chosen!r} but that file does not exist"
            )
        return path.resolve()
    for root in _ancestors(Path.cwd()):
        option = root / CONFIG_FILE
        if option.is_file():
            return option.resolve()
    return USER_CONFIG.resolve() if USER_CONFIG.is_file() else None


def load_config() -> tuple[dict[str, str], Path | None]:
    """Read the config file into absolute paths. Returns ({}, None) if absent."""
    path = config_path()
    if path is None:
        return {}, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must hold a JSON object, got {type(raw).__name__}")
    # A misspelled key would otherwise sit in the file looking configured while
    # the search quietly used something else.
    unknown = sorted(set(raw) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(
            f"{path} has unknown keys {unknown}; known keys are {list(CONFIG_KEYS)}"
        )
    values: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise ValueError(f"{path}: {key} must be a string path, got {type(value).__name__}")
        candidate = Path(value).expanduser()
        # A relative entry is relative to the config file itself, so a config
        # committed beside a checkout keeps working wherever the checkout goes.
        values[key] = str(candidate if candidate.is_absolute() else path.parent / candidate)
    return values, path


def _stated(flag: str, hint: Path | str | None, env_name: str, config_key: str
            ) -> list[tuple[str, str]]:
    """The ways this path was stated outright, highest precedence first."""
    config, config_file = load_config()
    stated = [
        (flag, None if hint is None else str(hint)),
        (env_name, os.environ.get(env_name)),
        (f"{config_key!r} in {config_file}", config.get(config_key)),
    ]
    return [(source, value) for source, value in stated if value]


def find_urdf_dir(hint: Path | str | None = None) -> Path:
    """Directory holding `x2_ultra.xml`, i.e. `.../X2_URDF-v1.3.0`.

    `hint` may be either that directory itself or the root of the vendor drop
    (the directory containing `aimdk/`), so `--urdf-dir /path/to/X2/test1` and
    `--urdf-dir /path/to/X2_URDF-v1.3.0` both work.
    """
    def _resolve(candidate: Path, tried: list[Path]) -> Path | None:
        candidate = Path(candidate).expanduser()
        for option in (candidate, candidate / URDF_SUBPATH):
            tried.append(option)
            if (option / MODEL_FILE).is_file():
                return option.resolve()
        return None

    for source, chosen in _stated("--urdf-dir", hint, ENV_URDF_DIR, "urdf_dir"):
        tried = []
        found = _resolve(Path(chosen), tried)
        if found is not None:
            return found
        raise FileNotFoundError(
            f"{source} was given as {chosen!r} but {MODEL_FILE} is not there. Looked in:\n  "
            + "\n  ".join(str(path) for path in tried)
        )

    tried = []
    for root in _search_roots():
        found = _resolve(root, tried)
        if found is not None:
            return found

    raise FileNotFoundError(
        f"could not find {MODEL_FILE}. Set {ENV_URDF_DIR} to the directory that "
        f"contains it (or to the parent of {URDF_SUBPATH}), or put \"urdf_dir\" "
        f"in a {CONFIG_FILE}. Tried:\n  "
        + "\n  ".join(str(path) for path in tried[:24])
    )


def _find_file(what: str, flag: str, hint: Path | str | None, env_name: str,
               config_key: str, subpath: Path,
               accept: "Callable[[Path], bool]" = Path.is_file) -> Path:
    """Locate one thing: stated outright, else the usual place up the tree.

    `accept` decides what counts as found, so the animation can also be a PEAR
    clip directory without the search itself learning anything about PEAR.
    """
    for source, chosen in _stated(flag, hint, env_name, config_key):
        path = Path(chosen).expanduser()
        if accept(path):
            return path.resolve()
        raise FileNotFoundError(f"{source} was given as {chosen!r} but that file does not exist")

    tried: list[Path] = []
    for root in _search_roots():
        option = root / subpath
        tried.append(option)
        if accept(option):
            return option.resolve()

    raise FileNotFoundError(
        f"could not find {what}. Pass {flag}, set {env_name}, or put "
        f"\"{config_key}\" in a {CONFIG_FILE}. Tried:\n  "
        + "\n  ".join(str(path) for path in tried[:24])
    )


def is_animation(path: Path | str) -> bool:
    """A canonical source-animation npz, or a PEAR clip directory."""
    path = Path(path)
    return path.is_file() or pear.is_clip_directory(path)


def find_animation(hint: Path | str | None = None) -> Path:
    """The source animation: a canonical .npz, or a PEAR clip directory."""
    return _find_file("the source animation", "--animation", hint,
                      ENV_ANIMATION, "animation", ANIMATION_SUBPATH,
                      accept=is_animation)


def find_ledger_source(hint: Path | str | None = None) -> Path:
    """The archive the packaged ledger was copied from.

    Only the drift check needs this. A checkout that cannot see the archive
    converts perfectly well; it just cannot check itself against it, and says so
    rather than reporting agreement it did not establish.
    """
    return _find_file("the ledger archive", "--ledger-source", hint,
                      ENV_LEDGER_SOURCE, "ledger_source", LEDGER_SOURCE_SUBPATH)


def model_path(urdf_dir: Path | str | None = None) -> Path:
    """The MJCF the kinematics loads. Forward kinematics needs nothing else."""
    return find_urdf_dir(urdf_dir) / MODEL_FILE


def scene_path(urdf_dir: Path | str | None = None) -> Path:
    """A render-ready scene wrapping the vendor MJCF, materialised on demand.

    MuJoCo resolves `<include>` and `meshdir` relative to the file they appear
    in, so a scene file cannot be vendored with a relative path into a tree it
    does not know the position of. It is instead written next to nothing, in a
    temp directory keyed by the resolved URDF path, with that path substituted
    absolute. Only `preview` needs this; the conversion pipeline does not.
    """
    resolved = find_urdf_dir(urdf_dir)
    cache = Path(tempfile.gettempdir()) / f"smplx2robot-scene-{abs(hash(str(resolved))):x}"
    cache.mkdir(parents=True, exist_ok=True)
    scene = cache / "x2_scene.xml"
    text = SCENE_TEMPLATE.read_text(encoding="utf-8").replace("{URDF_DIR}", str(resolved))
    if not scene.is_file() or scene.read_text(encoding="utf-8") != text:
        scene.write_text(text, encoding="utf-8")
    return scene
