from glob import glob

from setuptools import find_packages, setup

package_name = "robot_skill_cli"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/skills/ibrobot-control", ["resource/ibrobot-control/SKILL.md"]),
        (
            "share/" + package_name + "/hermes",
            [
                "resource/hermes/README.md",
                "resource/hermes/POLICY.md",
            ],
        ),
        (
            "share/" + package_name + "/hermes/hooks",
            [*glob("resource/hermes/hooks/*")],
        ),
    ],
    install_requires=["setuptools", "pyyaml"],
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Stable JSON command-line adapter for the IB-Robot Capability Gateway.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot-skill = robot_skill_cli.cli:main",
            "robot-skill-closed-loop = robot_skill_cli.closed_loop_runner:main",
            "hermes-robot = robot_skill_cli.hermes_launcher:main",
            "ibrobot-perceive = robot_skill_cli.perceive_cli:main",
        ]
    },
)
