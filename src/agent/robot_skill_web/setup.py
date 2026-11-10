from setuptools import find_packages, setup

package_name = "robot_skill_web"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    package_data={"robot_skill_web": ["static/*"]},
    include_package_data=True,
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "robot_client", "robot_config"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Local Web interface for the public IB-Robot client",
    license="Apache-2.0",
    entry_points={"console_scripts": ["robot-skill-web = robot_skill_web.app:main"]},
)
