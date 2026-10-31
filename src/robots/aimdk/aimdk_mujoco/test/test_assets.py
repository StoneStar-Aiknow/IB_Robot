"""Asset preparation with small public fixtures, without SDK or physics dependencies."""

import hashlib
import io
import json

import pytest
from aimdk_mujoco import assets


@pytest.fixture
def public(monkeypatch):
    resources = {
        "LICENSE": b"Mulan PSL v2 license fixture\n",
        "scene.xml": b'<mujoco><include file="x2_ultra.xml"/></mujoco>',
        "x2_ultra.xml": b'<mujoco model="public"><asset><mesh file="leg.STL"/></asset></mujoco>',
        "meshes/leg.STL": b"public mesh bytes",
    }
    remote = {"LICENSE" if name == "LICENSE" else f"X2_URDF-v1.3.0/{name}": data for name, data in resources.items()}
    tree = {
        "truncated": False,
        "tree": [
            {"path": name, "type": "blob", "sha": hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()}
            for name, data in remote.items()
        ],
    }
    responses = {f"{assets.SOURCE}/{name}": data for name, data in remote.items()}
    responses[assets.TREE_URL] = json.dumps(tree).encode()
    requests = []

    def urlopen(request, timeout):
        assert timeout > 0
        requests.append(request.full_url)
        return io.BytesIO(responses[request.full_url])

    monkeypatch.setattr(assets.urllib.request, "urlopen", urlopen)
    return resources, responses, requests


def test_transport_cache_uses_public_xml_and_standalone_checked_files(tmp_path, public):
    resources, _, requests = public
    cache = tmp_path / "sdk-robot"
    (cache / "meshes").mkdir(parents=True)
    (cache / "meshes/leg.STL").write_bytes(resources["meshes/leg.STL"])
    (cache / "x2_ultra.xml").write_bytes(b'<mujoco model="private"/>')
    destination = tmp_path / "assets"
    assets.prepare(destination, robot_cache=cache)
    assert f"{assets.SOURCE}/X2_URDF-v1.3.0/meshes/leg.STL" not in requests
    assert f"{assets.SOURCE}/X2_URDF-v1.3.0/x2_ultra.xml" in requests
    assert assets.TREE_URL in requests
    manifest = json.loads((destination / "assets.json").read_text())
    assert manifest["robot_revision"] == assets.REVISION
    assert manifest["robot_tree_api"] == assets.TREE_URL
    assert "MulanPSL-2.0" in manifest["robot_license"]
    assert set(manifest) == {
        "robot_source",
        "robot_revision",
        "robot_tree_api",
        "robot_license",
        "robot_git_blobs",
        "files",
    }
    assert not (destination / assets.PARTIAL).exists()
    assert set(manifest["files"]) == set(resources)
    for name, data in resources.items():
        target = destination / name
        assert target.read_bytes() == data
        assert not target.is_symlink()
        assert manifest["files"][name] == assets.digest(target)
        assert manifest["robot_git_blobs"][name] == hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
    assert not (destination / "policy.onnx").exists()
    assert {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()} == (
        set(resources) | {"assets.json"}
    )


def test_interrupted_download_leaves_resumable_marker(tmp_path, public):
    resources, responses, _ = public
    destination = tmp_path / "assets"
    mesh_url = f"{assets.SOURCE}/X2_URDF-v1.3.0/meshes/leg.STL"
    responses[mesh_url] = b"truncated"
    with pytest.raises(ValueError, match="integrity mismatch"):
        assets.prepare(destination)
    assert (destination / assets.PARTIAL).is_file()
    (destination / "meshes").mkdir()
    (destination / "meshes/leg.STL.part").write_bytes(b"interrupted staging file")
    responses[mesh_url] = resources["meshes/leg.STL"]
    assets.prepare(destination, resume=True)
    assert (destination / "meshes/leg.STL").read_bytes() == resources["meshes/leg.STL"]
    assert not list(destination.rglob("*.part"))
    assert not (destination / assets.PARTIAL).exists()
