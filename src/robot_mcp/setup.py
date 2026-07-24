from setuptools import find_packages, setup

package_name = "robot_mcp"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/robot_mcp.launch.py"]),
    ],
    install_requires=["setuptools", "mcp>=1.26.0,<2"],
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="MCP tool layer exposing IB-Robot runtime skills/status to MCP-compatible agent hosts.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "robot_mcp_server = robot_mcp.server:main",
        ],
    },
)
