#!/usr/bin/env python3
"""Materialise src/robots/aimdk/aimdk_description from the official AgiBot X2 URDF.

Source: https://github.com/AgibotTech/agibot_x2_urdf at a pinned commit, licensed
under the Mulan Permissive Software License v2. The package keeps upstream files
byte-for-byte and records each one's SHA-256; nothing is edited by hand. Even
with --source, every byte is read from the pinned commit's raw git objects,
never from that checkout's working tree or its local git configuration, so
neither local edits nor machine-specific filters can ship as upstream.

What is taken, and why:
  * X2_URDF-v1.3.0/x2_ultra.urdf - the X2 Ultra (non "-N") body. Its kinematics
    match the motion-control unit's own model on the verified robot exactly.
  * X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf - the only official OmniHand model.
    v1.3.0 ships no hand; the v1.3.0 and v1.4.0 wrist frames agree within a
    millimetre, so the hand is reused on the v1.3.0 arm by later tooling.
  * The meshes of the upper body (pelvis, waist, torso, both arms, head) and of
    the left OmniHand, plus the v1.4.0 wrist roll links: the v1.3.0 ones carry
    the dummy hand fitted when no OmniHand is mounted. Legs and IMU/camera
    housings are not needed for upper-body planning and are left out to keep
    the package small.

Usage:
    scripts/sync_x2_description.py            # clone the pinned commit and copy it
    scripts/sync_x2_description.py --source D  # a checkout of the pinned commit; its working tree is
                                               # ignored, only the commit's git objects are read
    scripts/sync_x2_description.py --check     # verify the package against the manifest
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

REPOSITORY = "https://github.com/AgibotTech/agibot_x2_urdf.git"
COMMIT = "575cc6b988f976c23550e0db85aa1e5475d3652d"
PACKAGE = Path(__file__).resolve().parents[1] / "src" / "robots" / "aimdk" / "aimdk_description"

BODY_URDF = "X2_URDF-v1.3.0/x2_ultra.urdf"
HAND_URDF = "X2_URDF-v1.4.0/X2-Ultra_omnihand.urdf"
#: Body links whose meshes are shipped: the upper body, and the left leg the grasp
#: planner keeps the left hand away from. The right leg, the feet and the IMU /
#: camera housings are left out.
BODY_LINKS = (
    "pelvis",
    *(f"left_{part}_link" for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee")),
    "waist_yaw_link",
    "waist_pitch_link",
    "torso_link",
    *(
        f"{side}_{part}_link"
        for side in ("left", "right")
        for part in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_yaw",
            "wrist_pitch",
            "wrist_roll",
        )
    ),
    "head_yaw_link",
    "head_pitch_link",
)
HAND_LINK_PREFIX = "L_"
#: OmniHand-side links of the v1.4.0 model shipped besides the hand itself.
HAND_MOUNT_LINKS = ("left_wrist_roll_link", "right_wrist_roll_link")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _meshes(urdf: Path, keep) -> list[str]:
    root = ET.parse(urdf).getroot()
    names = set()
    for link in root.findall("link"):
        if keep(link.get("name")):
            names.update(Path(mesh.get("filename")).name for mesh in link.iter("mesh"))
    return sorted(names)


def _selection(upstream: Path) -> dict[str, str]:
    """Map each package path to the upstream file it is materialised from."""
    files = {
        "LICENSE": "LICENSE",
        "urdf/x2_ultra.urdf": BODY_URDF,
        "urdf/x2_ultra_omnihand_v1.4.0.urdf": HAND_URDF,
    }
    for mesh in _meshes(upstream / BODY_URDF, lambda name: name in BODY_LINKS):
        files[f"meshes/v1.3.0/{mesh}"] = f"X2_URDF-v1.3.0/meshes/{mesh}"
    for mesh in _meshes(
        upstream / HAND_URDF, lambda name: name.startswith(HAND_LINK_PREFIX) or name in HAND_MOUNT_LINKS
    ):
        files[f"meshes/v1.4.0/{mesh}"] = f"X2_URDF-v1.4.0/meshes/{mesh}"
    return files


#: First line of a Git LFS pointer blob, which stands in for the real content.
POINTER_VERSION = "version https://git-lfs/spec/v1"


def _lfs_pointer(blob: bytes) -> tuple[str, int] | None:
    """Return (oid, size) when `blob` is a genuine Git LFS pointer, else None."""
    try:
        lines = blob.decode("utf-8").splitlines()
    except UnicodeDecodeError:
        return None
    if not lines or lines[0] != POINTER_VERSION:
        return None
    fields = {}
    for line in lines:
        key, sep, value = line.partition(" ")
        if sep and key not in fields:
            fields[key] = value
    prefix = "sha256:"
    oid = fields.get("oid", "")
    size = fields.get("size", "")
    if not oid.startswith(prefix) or len(oid) != len(prefix) + 64 or not size.isdigit():
        return None
    return oid[len(prefix) :], int(size)


def _git_dirs(source: Path) -> list[Path]:
    """The absolute git dir and common dir of the repository at `source`."""
    dirs: list[Path] = []
    for flag in ("--git-common-dir", "--git-dir"):
        out = subprocess.run(
            ["git", "-C", str(source), "rev-parse", flag], check=True, capture_output=True, text=True
        ).stdout.strip()
        if not out:
            continue
        path = Path(out)
        if not path.is_absolute():
            path = source / path
        if path not in dirs:
            dirs.append(path)
    return dirs


def _lfs_content(source: Path, pointer: tuple[str, int], origin: str) -> bytes:
    """Resolve a Git LFS pointer blob to its content from the local object store.

    Checkout filters are deliberately not used: their configuration belongs to
    one machine. The object must carry the pointer's recorded SHA-256 and size,
    so the bytes are exactly the ones the pinned commit references.
    """
    oid, size = pointer
    for git_dir in _git_dirs(source):
        path = git_dir / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
        if path.is_file():
            content = path.read_bytes()
            if len(content) != size or hashlib.sha256(content).hexdigest() != oid:
                sys.exit(f"{origin}: Git LFS object {oid[:12]} does not match its pointer")
            return content
    sys.exit(f"{origin}: Git LFS object {oid[:12]} is missing; run 'git lfs fetch' in {source}")


def _export_file(source: Path, target: Path, origin: str) -> None:
    """Copy `origin`'s raw blob from COMMIT, resolving only genuine LFS pointers."""
    blob = subprocess.run(
        ["git", "-C", str(source), "cat-file", "blob", f"{COMMIT}:{origin}"], check=True, capture_output=True
    ).stdout
    pointer = _lfs_pointer(blob)
    if pointer is not None:
        blob = _lfs_content(source, pointer, origin)
    out = target / origin
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)


def _export(source: Path, target: Path) -> None:
    """Materialise the selection into `target` straight from COMMIT's raw blobs.

    `source` is a git repository holding COMMIT's objects -- either the fresh
    shallow fetch or the user's `--source` checkout. That checkout may carry
    staged or unstaged edits plus local git configuration; neither its
    working-tree bytes nor the filters and line-ending rules of one machine may
    be hashed into a manifest that stamps the official commit. Every file
    therefore comes from `git cat-file blob COMMIT:<path>`, and Git LFS pointer
    blobs are resolved from the source repository's local object store with
    their recorded SHA-256 and size verified.
    """
    for origin in (BODY_URDF, HAND_URDF):
        _export_file(source, target, origin)  # the URDFs go first: _selection parses them
    for origin in _selection(target).values():
        _export_file(source, target, origin)


def _checkout(source: Path | None, workdir: Path) -> Path:
    """Return a pristine snapshot of the pinned commit, never a live working tree."""
    target = workdir / "agibot_x2_urdf"
    if source is None:
        subprocess.run(["git", "init", "--quiet", str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "fetch", "--quiet", "--depth", "1", REPOSITORY, COMMIT], check=True)
        source = target
    else:
        head = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        if head != COMMIT:
            sys.exit(f"{source} is at {head}, expected {COMMIT}")
    _export(source, target)
    return target


def sync(source: Path | None) -> None:
    with tempfile.TemporaryDirectory() as workdir:
        upstream = _checkout(source, Path(workdir))
        files = _selection(upstream)
        for directory in ("urdf", "meshes"):
            shutil.rmtree(PACKAGE / directory, ignore_errors=True)
        manifest = []
        for target, origin in sorted(files.items()):
            (PACKAGE / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(upstream / origin, PACKAGE / target)
            manifest.append({"path": target, "upstream": origin, "sha256": _sha256(PACKAGE / target)})
    document = {
        "upstream": {"repository": REPOSITORY, "commit": COMMIT, "license": "MulanPSL-2.0"},
        "files": manifest,
    }
    (PACKAGE / "manifest.yaml").write_text(
        "# Generated by scripts/sync_x2_description.py; do not edit.\n" + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    print(f"synced {len(manifest)} files from {REPOSITORY}@{COMMIT[:7]}")


def check() -> int:
    document = yaml.safe_load((PACKAGE / "manifest.yaml").read_text(encoding="utf-8"))
    bad = [
        entry["path"]
        for entry in document["files"]
        if not (PACKAGE / entry["path"]).is_file() or _sha256(PACKAGE / entry["path"]) != entry["sha256"]
    ]
    if bad:
        print(f"{len(bad)} files differ from the manifest: {bad[:5]}")
        return 1
    print(
        f"{len(document['files'])} files match {document['upstream']['repository']}@{document['upstream']['commit'][:7]}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--source",
        type=Path,
        help=f"existing checkout of {REPOSITORY} at {COMMIT[:7]}; only the commit's raw git objects are read",
    )
    parser.add_argument("--check", action="store_true", help="verify the package against its manifest")
    args = parser.parse_args()
    if args.check:
        return check()
    sync(args.source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
