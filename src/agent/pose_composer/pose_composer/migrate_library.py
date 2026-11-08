"""Re-stamp a fixed pose library with the motion-only robot fingerprint; no MuJoCo needed.

Libraries compiled before ``aimdk_grasp.fixed_pose_backend.FINGERPRINT_SCHEME``
carry a fingerprint over the whole runtime profile, so an unrelated profile
change (an audio channel count) rejects them. Migration changes no pose: it
requires the library's stored fingerprint to equal the legacy fingerprint of
exactly the given profile, limits and hand model, proving the library was
compiled for them, then rewrites only ``robot_fingerprint`` and the digest.
After that, a library keeps loading for as long as the motion inputs match.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml
from aimdk_grasp.fixed_pose_backend import FixedBackend, digest


def restamp(data: dict, backend: FixedBackend) -> tuple[dict, bool]:
    """``data`` re-stamped for ``backend`` and whether anything changed. Poses are untouched."""
    payload = {key: value for key, value in data.items() if key != "digest"}
    if data.get("schema_version") != 1 or data.get("digest") != digest(payload):
        raise ValueError("invalid compiled pose library digest/version")
    stored = data.get("robot_fingerprint")
    if stored == backend.fingerprint:
        return data, False
    if stored != backend.legacy_fingerprint:
        raise ValueError(
            "the library was not compiled for this runtime profile, these arm limits and this hand model: "
            "recompile it with pose_compile_library"
        )
    payload["robot_fingerprint"] = backend.fingerprint
    return {**payload, "digest": digest(payload)}, True


def migrate(data: dict, profile, limits=None, *, task_poses=None) -> tuple[dict, bool]:
    migrated, changed = restamp(data, FixedBackend(profile, limits))
    # The result must load exactly as the live executor loads it.
    FixedBackend(profile, limits, migrated, task_poses=task_poses)
    return migrated, changed


def robot_config_inputs(config_path) -> dict:
    """The profile, limits, task poses and library a live fixed-pose launch of this robot uses."""
    from robot_config.fixed_poses import fixed_library_path
    from robot_config.loader import _load_robot_section
    from robot_config.task_poses import resolve_task_pose_file
    from robot_runtime.launch_support import resolve_profile_argument

    path, config = _load_robot_section(config_path)
    if (config.get("runtime") or {}).get("provider") != "aimdk_robot":
        raise ValueError("fixed X2 libraries require runtime.provider: aimdk_robot")
    if (config.get("pose_composer") or {}).get("mode") != "fixed":
        raise ValueError("--robot-config requires pose_composer.mode: fixed")
    resolve_task_pose_file(config, path)
    return {
        "profile": resolve_profile_argument("aimdk_robot", config["runtime"]["profile"]),
        "limits": config.get("arm_limits"),
        "task_poses": config.get("task_poses"),
        "library": fixed_library_path(config, path),
    }


def _write_new(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-config", help="robot YAML; derives the profile, arm limits, task poses and library")
    parser.add_argument("--profile", help="runtime profile the library was compiled with")
    parser.add_argument("--limits", help="YAML mapping of all 14 arm limits the library was compiled with")
    parser.add_argument("--library", help="compiled fixed pose library")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", help="write the migrated library here (must not exist)")
    target.add_argument(
        "--in-place", action="store_true", help="replace the library; the original is kept as *.legacy-fingerprint.json"
    )
    args = parser.parse_args(argv)
    if args.robot_config:
        if args.profile or args.limits:
            parser.error("--robot-config cannot be combined with --profile or --limits")
        inputs = robot_config_inputs(args.robot_config)
        if args.library:
            inputs["library"] = Path(args.library)
    else:
        if not (args.profile and args.library):
            parser.error("provide --robot-config, or --profile and --library")
        limits = yaml.safe_load(Path(args.limits).read_text()) if args.limits else None
        inputs = {"profile": args.profile, "limits": limits, "task_poses": None, "library": Path(args.library)}
    library = Path(inputs["library"])
    original = library.read_text()
    migrated, changed = migrate(
        json.loads(original), inputs["profile"], inputs["limits"], task_poses=inputs["task_poses"]
    )
    if not changed:
        print(f"{library} already carries the motion-only fingerprint; nothing written")
        return
    if args.in_place:
        backup = library.with_name(library.stem + ".legacy-fingerprint.json")
        with backup.open("x") as stream:
            stream.write(original)
        staged = library.with_name(f".{library.name}.migrating")
        _write_new(staged, migrated)
        os.replace(staged, library)
        written = library
        print(f"original kept as {backup}")
    else:
        written = Path(args.output)
        _write_new(written, migrated)
    print(f"migrated {written}: robot_fingerprint={migrated['robot_fingerprint']} digest={migrated['digest']}")


if __name__ == "__main__":
    main()
