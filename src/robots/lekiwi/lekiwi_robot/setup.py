from glob import glob

from setuptools import find_packages, setup

package_name = "lekiwi_robot"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/profiles", glob("profiles/*.yaml")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="LeKiwi robot runtime: standalone launch of the complete LeKiwi stack from a runtime profile",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "base_node = lekiwi_robot.base_node:main",
            "fast_lio_odom_bridge = lekiwi_robot.fast_lio_odom_bridge:main",
        ],
    },
)
