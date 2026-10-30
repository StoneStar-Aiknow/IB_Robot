from glob import glob

from setuptools import find_packages, setup

package_name = "aimdk_mujoco"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "THIRD_PARTY.md", "LICENSE"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "mujoco>=3.0", "numpy", "onnxruntime>=1.16"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Standalone X2 standing and walking simulation with MuJoCo and standard ROS messages",
    license="Apache-2.0",
    entry_points={"console_scripts": ["simulation = aimdk_mujoco.node:main"]},
)
