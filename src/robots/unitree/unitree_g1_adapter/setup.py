from setuptools import find_packages, setup

package_name = "unitree_g1_adapter"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    # Deliberately empty: unitree_sdk2py is not on PyPI. It is installed into
    # the workspace venv at a pinned revision by
    # scripts/setup/install_unitree_sdk.sh, and this adapter import-guards it
    # (unitree_g1_adapter.vendor) instead of declaring an unresolvable dependency.
    install_requires=["setuptools"],
    # colcon test falls back to unittest (0 tests, false green) without this.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="IB-Robot Team",
    maintainer_email="maintainer@example.com",
    description="ROS-free Unitree G1 SDK adapter over the official unitree_sdk2_python",
    license="Apache-2.0",
)
