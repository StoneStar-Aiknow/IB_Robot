from setuptools import find_packages, setup

package_name = "pose_composer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # mujoco is a venv dependency of the workspace (requirements); the model is the
    # vendor's, fetched by aimdk_robot's aimdk_fetch_x2_mjcf, never vendored.
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Natural-language arm poses for the AgiBot X2",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pose_composer_node = pose_composer.node:main",
        ],
        "ibrobot.launch_builders": [
            "pose_composer = pose_composer.launch_builder:POSE_COMPOSER_BUILDER",
        ],
    },
)
