from glob import glob

from setuptools import find_packages, setup

package_name = "manipulation_execution"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/data/x2_mock_imitation",
            glob("data/x2_mock_imitation/*"),
        ),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Contributors",
    maintainer_email="roboguru.92@gmail.com",
    description="Closed-loop manipulation execution for agent-facing robot skills",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_action_client = manipulation_execution.pick_action_client:main",
            "pick_executor_node = manipulation_execution.pick_executor_node:main",
            "place_executor_node = manipulation_execution.placement_executor_node:main",
            "imitate_human_motion_executor_node = manipulation_execution.imitate_human_motion_executor_node:main",
            "motion_imitation_stream_node = manipulation_execution.motion_imitation_stream_node:main",
            "motion_imitation_mock_node = manipulation_execution.motion_imitation_mock_node:main",
            "mirror_session_node = manipulation_execution.mirror_session_node:main",
            "mirror_session_release = manipulation_execution.mirror_session_node:release_main",
            "placement_replay = manipulation_execution.placement_replay:main",
        ],
        # Cross-package launch-builder registration (openspec change
        # ``decentralize-launch-builder-registration``, design decision 4):
        # robot_config discovers this group when building the launch registry,
        # so the mirror session's launch wiring registers from its owning
        # package without any robot_config change. Installing this package
        # does not enable the session; participation still comes from the
        # robot YAML (``mirror_session.enabled``). Openspec change
        # ``migrate-business-launch-builders`` additionally moved the module
        # to the unified ``launch_builders/`` layout and added the
        # motion-imitation stream entry.
        "ibrobot.launch_builders": [
            "mirror_session = manipulation_execution.launch_builders.mirror_session:MIRROR_SESSION_BUILDER",
            "motion_imitation_stream = manipulation_execution.launch_builders.motion_imitation_stream:MOTION_IMITATION_STREAM_BUILDER",
        ],
    },
)
