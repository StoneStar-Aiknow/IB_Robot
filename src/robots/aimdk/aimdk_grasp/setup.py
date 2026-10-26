from glob import glob

from setuptools import find_packages, setup

package_name = "aimdk_grasp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/x2_ultra", glob("config/x2_ultra/*")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="weixin_54906338",
    maintainer_email="zhufei26@huawei.com",
    description="AgiBot X2 grasp pipeline over the public runtime interfaces",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "fit_x2_torso_cover = aimdk_grasp.torso_cover:main",
            "export_x2_leg_obstacle = aimdk_grasp.leg_obstacle:main",
        ],
    },
)
