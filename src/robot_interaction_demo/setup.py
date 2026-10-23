from setuptools import find_packages, setup

package_name = "robot_interaction_demo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Operator-triggered speech and gesture business examples over public runtime APIs",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "interaction_demo_node = robot_interaction_demo.node:main",
            "runtime-demo = robot_interaction_demo.client:main",
        ],
        # Cross-package launch-builder registration (openspec change
        # ``migrate-business-launch-builders``): robot_config discovers this
        # group when building the launch registry, so the interaction demo's
        # launch wiring registers from its owning package without any
        # robot_config change. Installing this package does not enable the
        # entry; participation still comes from the robot YAML
        # (``interaction_demo.enabled``).
        "ibrobot.launch_builders": [
            "interaction_demo = robot_interaction_demo.launch_builders.interaction_demo:INTERACTION_DEMO_BUILDER",
        ],
    },
)
