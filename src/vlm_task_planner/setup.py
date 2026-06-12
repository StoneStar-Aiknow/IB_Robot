from setuptools import find_packages, setup

package_name = "vlm_task_planner"

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
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="VLM-backed task planner for the embodied execution pipeline",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vlm_task_planner_node = vlm_task_planner.vlm_task_planner_node:main",
        ],
    },
)
