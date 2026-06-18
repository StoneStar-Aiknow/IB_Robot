from setuptools import find_packages, setup

package_name = "grasp_service"

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
    description="Grasp planning service: GraspGen for IB-Robot",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grasp_planner_node = grasp_service.grasp_planner_node:main",
        ],
    },
)
