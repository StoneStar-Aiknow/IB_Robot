"""Download and verify public robot assets independently of policy weights."""

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REVISION = "77f43eb0904dae4c48ccd9154fee824f8ffd4d38"
SOURCE = f"https://raw.githubusercontent.com/AgibotTech/agibot_x2_urdf/{REVISION}"
TREE_URL = f"https://api.github.com/repos/AgibotTech/agibot_x2_urdf/git/trees/{REVISION}?recursive=1"
POLICY_SHA256 = "5ca4ebc3ae12e902cb4346ecc4b8532e3a2404a41f0a56e631a2d0316e297a71"
PARTIAL = ".assets.partial.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_blob(data):
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def prepare(destination, robot_cache=None, resume=False):
    if destination.is_symlink():
        raise FileExistsError(f"Refusing a symlink asset cache: {destination}")
    destination = destination.resolve()
    if (destination / "assets.json").exists() or (destination / "assets.json").is_symlink():
        raise FileExistsError(f"Refusing to overwrite a completed asset cache: {destination}")
    if destination.exists() and (not resume or not destination.is_dir()):
        raise FileExistsError(
            f"Refusing to overwrite an existing asset cache: {destination}; use --resume for partials"
        )
    if robot_cache is not None:
        robot_cache = robot_cache.resolve(strict=True)
        if not robot_cache.is_dir():
            raise NotADirectoryError(robot_cache)

    def download(url):
        request = urllib.request.Request(url, headers={"User-Agent": "aimdk-mujoco-assets"})
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    tree = json.loads(download(TREE_URL))
    if tree.get("truncated", True):
        raise ValueError("Public Git tree is truncated or missing its completeness flag")
    blobs = {entry["path"]: entry["sha"] for entry in tree["tree"] if entry["type"] == "blob"}
    remotes = {
        "LICENSE": "LICENSE",
        "scene.xml": "X2_URDF-v1.3.0/scene.xml",
        "x2_ultra.xml": "X2_URDF-v1.3.0/x2_ultra.xml",
    }

    def verify(relative, data):
        expected = blobs[remotes[relative]]
        if _git_blob(data) != expected:
            raise ValueError(f"Git blob integrity mismatch for {relative} at {REVISION}")
        return data

    # XML always comes from the pinned public source, never from the SDK tree.
    roots = {name: verify(name, download(f"{SOURCE}/{remote}")) for name, remote in remotes.items()}
    root = ET.fromstring(roots["x2_ultra.xml"])
    mesh_names = {mesh.attrib["file"] for mesh in root.findall("asset/mesh")}
    if any(Path(name).name != name or name in ("", ".", "..") or "\\" in name for name in mesh_names):
        raise ValueError("Public XML contains an unsafe mesh path")
    meshes = sorted(f"meshes/{name}" for name in mesh_names)
    remotes.update({name: f"X2_URDF-v1.3.0/{name}" for name in meshes})
    if missing := set(remotes.values()) - blobs.keys():
        raise ValueError(f"Public Git tree is missing required assets: {sorted(missing)}")

    marker = destination / PARTIAL
    partial = {"robot_source": "https://github.com/AgibotTech/agibot_x2_urdf", "robot_revision": REVISION}
    verified = {}
    if destination.exists():
        marked = marker.is_file() and not marker.is_symlink()
        if marked and json.loads(marker.read_text()) != partial:
            raise ValueError(f"Partial cache marker does not match the public revision: {marker}")
        allowed = set(remotes)
        if marked:
            allowed |= {PARTIAL, "assets.json.part"} | {f"{name}.part" for name in allowed}
        entries = list(destination.rglob("*"))
        for path in entries:
            name = path.relative_to(destination).as_posix()
            if path.is_symlink() or not (path.is_dir() and name == "meshes" or path.is_file() and name in allowed):
                raise FileExistsError(f"Refusing unknown asset cache entry: {path}")
            if path.is_file() and name in remotes:
                data = path.read_bytes()
                # The old downloader wrote directly to meshes, leaving truncated files on timeout.
                if name in meshes and _git_blob(data) != blobs[remotes[name]]:
                    continue
                verify(name, data)
                verified[name] = hashlib.sha256(data).hexdigest()
        # Adopt the old downloader's unmarked partial only with all three exact public roots.
        if not marked and not roots.keys() <= verified.keys():
            raise FileExistsError(f"Refusing an unrecognized partial asset cache: {destination}")
        for path in entries:
            if path.name.endswith(".part"):
                path.unlink()
    else:
        destination.mkdir(parents=True)
    if not marker.exists():
        with marker.open("x") as output:
            output.write(json.dumps(partial, indent=2) + "\n")

    def install(relative, data):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".part")
        try:
            with temporary.open("xb") as output:
                output.write(data)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
        return relative, hashlib.sha256(data).hexdigest()

    def fetch(relative):
        if relative in verified:
            return relative, verified[relative]
        cached = robot_cache / relative if robot_cache is not None else None
        if cached is not None and cached.is_file():
            data = cached.read_bytes()
        else:
            data = download(f"{SOURCE}/{remotes[relative]}")
        return install(relative, verify(relative, data))

    files = dict(verified)
    for name, data in roots.items():
        if name not in verified:
            files.update([install(name, data)])
    with ThreadPoolExecutor(max_workers=4) as executor:
        files.update(executor.map(fetch, meshes))
    manifest = {
        **partial,
        "robot_tree_api": TREE_URL,
        "robot_git_blobs": {name: blobs[remote] for name, remote in remotes.items()},
        "robot_license": "MulanPSL-2.0 (see LICENSE)",
        "files": files,
    }
    install("assets.json", (json.dumps(manifest, indent=2) + "\n").encode())
    marker.unlink()
    print(f"Prepared {len(files)} checked assets in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True, help="Destination directory for prepared assets")
    parser.add_argument(
        "--robot-cache",
        type=Path,
        help="Optional X2_URDF-v1.3.0 mesh transport cache; checked against public Git blobs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a verified, incomplete asset cache")
    options = parser.parse_args()
    prepare(options.assets, robot_cache=options.robot_cache, resume=options.resume)


if __name__ == "__main__":
    main()
