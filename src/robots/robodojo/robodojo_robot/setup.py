from glob import glob

from setuptools import find_packages, setup

package_name = "robodojo_robot"

setup(
    name=package_name,
    version="0.1.0",
    # robodojo_client/ runs inside RoboDojo's Isaac process (Python 3.11) through a
    # XPolicyLab wrapper; it is not part of the ROS package.
    packages=find_packages(exclude=["test", "robodojo_client"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/profiles", glob("profiles/*.yaml")),
        # Isaac-side files (Python 3.11, no ROS): linked into RoboDojo's XPolicyLab by run_eval_client.sh.
        (
            "share/" + package_name + "/robodojo_client",
            glob("robodojo_client/*.py") + glob("robodojo_client/*.sh") + glob("robodojo_client/*.yml"),
        ),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="leiziting",
    maintainer_email="leiziting@huawei.com",
    description="RoboDojo dual-arm ARX X5 as an IB-Robot runtime (virtual robot body)",
    license="Apache-2.0",
    entry_points={
        # The robodojo simulation platform (sim-platform-registry D1). Its embodiment starts
        # robot_runtime_sim's direct-control runtime with this package's adapter.
        "ibrobot.sim_platforms": [
            "robodojo = robodojo_robot.platforms:ROBODOJO_SPEC",
        ],
    },
)
