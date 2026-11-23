from glob import glob

from setuptools import find_packages, setup

package_name = "agent_eval_robodojo"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/suites", glob("config/suites/*.yaml")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="leiziting",
    maintainer_email="leiziting@huawei.com",
    description="RoboDojo platform plugin and suites for agent_eval",
    license="Apache-2.0",
    entry_points={
        "agent_eval.platforms": [
            "robodojo = agent_eval_robodojo.platform:RoboDojoPlatform",
        ],
    },
)
