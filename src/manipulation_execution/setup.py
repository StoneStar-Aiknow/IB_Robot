from setuptools import find_packages, setup

package_name = "manipulation_execution"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="IB-Robot Contributors",
    maintainer_email="roboguru.92@gmail.com",
    description="Closed-loop grasp execution for agent-facing robot skills",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "pick_executor_node = manipulation_execution.pick_executor_node:main",
        ],
    },
)
