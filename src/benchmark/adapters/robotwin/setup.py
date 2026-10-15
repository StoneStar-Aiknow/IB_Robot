from setuptools import setup

package_name = "benchmark_robotwin"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name, ["pytest.ini"]),
    ],
    install_requires=["setuptools", "numpy"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="ib-robot",
    maintainer_email="ib-robot@example.com",
    description="RoboTwin benchmark adapter plugin for IB-Robot benchmark runtime.",
    license="Apache-2.0",
    entry_points={
        "ibrobot.benchmark_adapters": [
            "robotwin = benchmark_robotwin.plugin:create_plugin",
        ],
    },
)
