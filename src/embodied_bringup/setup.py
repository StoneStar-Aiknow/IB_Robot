from setuptools import find_packages, setup

package_name = "embodied_bringup"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/embodied_pipeline.launch.py"]),
        (
            "share/" + package_name + "/sros2",
            ["sros2/caller_policy.xml", "sros2/governance.xml", "sros2/README.md"],
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Launch orchestration for the embodied AI runtime pipeline",
    license="Apache-2.0",
)
