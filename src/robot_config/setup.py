from glob import glob
from pathlib import Path

from setuptools import find_packages, setup

package_name = "robot_config"


def _package_data_files(base_dir: str):
    entries = []
    for path in sorted(Path(base_dir).rglob("*")):
        if not path.is_file():
            continue
        install_dir = f"share/{package_name}/{path.parent.as_posix()}"
        entries.append((install_dir, [str(path)]))
    return entries


def _rt_infer_payload_files():
    """Install the RT-Infer vendor bundle payload as robot_config share data.

    The launch owner installs the bundle to /opt/rt-infer/<version> through
    the vendor's own install command; the payload it reads must therefore
    be available from an installed workspace, not only a source tree. Only
    the files the bundle's manifest declares are shipped (artifacts, payload
    files, metadata), mirroring exactly what the vendor install copies.
    """
    import json
    import os

    root = Path(__file__).resolve().parent.parent.parent
    package_root = Path(__file__).resolve().parent
    bundle_root = root / "third_party" / "vendor" / "rt-infer"
    entries = []
    for manifest_path in sorted(bundle_root.glob("*/manifest.json")):
        version = manifest_path.parent.name
        bundle = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        names = ["manifest.json"]
        for artifact in manifest.get("artifacts", {}).values():
            names.append(artifact["path"])
        for entry in manifest.get("payload_files", []):
            names.append(entry["path"])
        names.extend(manifest.get("metadata_files", []))
        # setup() resolves source paths against the setup.py directory.
        vendor_rel = Path(os.path.relpath(bundle_root, package_root))
        for name in names:
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Unsafe RT-Infer payload path: {name}")
            source = bundle / name
            if not source.is_file():
                raise ValueError(f"Missing RT-Infer payload: {source}")
            relative = source.relative_to(bundle)
            target_dir = f"share/{package_name}/rt_infer/{version}/{relative.parent.as_posix()}"
            entries.append((target_dir, [vendor_rel.joinpath(version, relative).as_posix()]))
    return entries


setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/test/config", glob("test/config/*.yaml") + glob("test/config/*.urdf.xacro")),
    ]
    + _package_data_files("config")
    + _rt_infer_payload_files(),
    install_requires=["setuptools", "pyyaml", "inference_manifest"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Unified robot configuration system for ros2_control and peripherals",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "ibrobot-trace-topology = robot_config.tracing_topology:main",
            "wait_for_clock = robot_config.wait_for_clock:main",
            "wait_for_controllers = robot_config.wait_for_controllers:main",
            "controller_spawner = robot_config.controller_spawner:main",
            "topic_relay = robot_config.topic_relay:main",
            "semantic_mapping_recorder = robot_config.semantic_mapping_recorder:main",
            "rt_infer_scheduler_supervisor = robot_config.rt_infer_scheduler_supervisor:main",
        ],
    },
)
