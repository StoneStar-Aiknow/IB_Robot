from setuptools import find_packages, setup

package_name = "dataset_tools"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    package_data={
        "dataset_tools.camera_isp": [
            "lut_data.npz",
            "camera_isp_offline_tables.json",
        ],
    },
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "av", "pandas", "numpy", "scipy", "matplotlib", "pyarrow"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Dataset collection and conversion tools for Imitation Learning",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_alignment = dataset_tools.camera_alignment:main",
            "camera_isp_calibrator = dataset_tools.camera_isp_calibrator:main",
            "episode_recorder = dataset_tools.episode_recorder:main",
            "bag_to_lerobot = dataset_tools.bag_to_lerobot:main",
            "policy_eval = dataset_tools.policy_eval:main",
            "record_cli = dataset_tools.record_cli:main",
            "frame_detector = dataset_tools.frame_detector:main",
            "rerun_viewer = dataset_tools.rerun_viewer:main",
            "lerobot_action_gap_repair = dataset_tools.lerobot_action_gap_repair:main",
        ],
        # Cross-package launch-builder registration (openspec change
        # ``migrate-business-launch-builders``): robot_config discovers this
        # group when building the launch registry, so the recording and rerun
        # visualizer launch wiring registers from its owning package without
        # any robot_config change. Installing this package does not enable
        # the entries; participation still comes from the record /
        # record_visualizer launch arguments.
        "ibrobot.launch_builders": [
            "recording = dataset_tools.launch_builders.recording:RECORDING_BUILDER",
            "rerun_visualizer = dataset_tools.launch_builders.recording:RERUN_VISUALIZER_BUILDER",
        ],
    },
)
