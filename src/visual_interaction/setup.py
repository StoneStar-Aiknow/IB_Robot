from setuptools import find_packages, setup

package_name = "visual_interaction"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    # Load-bearing: without the pytest extra colcon falls back to unittest,
    # collects nothing and reports a false green (precedent: commit 0b164ff3).
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Contributors",
    maintainer_email="roboguru.92@gmail.com",
    description="Proactive visual interaction: watch the visitor, decide what is worth answering, and say it once",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "visual_interaction_node = visual_interaction.visual_interaction_node:main",
        ],
        # Cross-package launch-builder registration: robot_config discovers
        # this group when building the launch registry, so the wiring registers
        # from its owning package without any robot_config change. Installing
        # this package does not enable the entry; participation still comes
        # from the robot YAML (``visual_interaction.enabled``).
        "ibrobot.launch_builders": [
            "visual_interaction = visual_interaction.launch_builders.visual_interaction:VISUAL_INTERACTION_BUILDER",
        ],
    },
)
