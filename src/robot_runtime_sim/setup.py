from glob import glob

from setuptools import find_packages, setup

package_name = "robot_runtime_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    # The toy profile the direct-control test doubles load (robot_runtime_sim.testing).
    package_data={"robot_runtime_sim.testing": ["*.yaml"]},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/worlds", glob("worlds/*.sdf")),
    ],
    install_requires=["setuptools", "pyyaml"],
    # colcon test falls back to unittest (0 tests, false green) without this.
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Simulation platform providers for the robot runtime: Gazebo and MuJoCo embodiments, platform specs, sim-only nodes and the direct-control runtime for platforms without a controller",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "wait_for_clock = robot_runtime_sim.wait_for_clock:main",
            "joint_current_publisher = robot_runtime_sim.joint_current_publisher:main",
            "finite_cloud_filter = robot_runtime_sim.finite_cloud_filter:main",
            "imu_settle_filter = robot_runtime_sim.imu_settle_filter:main",
            "odometry_twist_relay = robot_runtime_sim.odometry_twist_relay:main",
            # The runtime of a platform without a controller (direct_control); a platform's
            # embodiment starts it with the platform's adapter.
            "direct_control_runtime = robot_runtime_sim.direct_control.node:main",
        ],
        # The physics platforms this package provides (openspec
        # sim-platform-registry D1/D6); a platform is discovered through its
        # ibrobot.sim_platforms entry point.
        "ibrobot.sim_platforms": [
            "gazebo = robot_runtime_sim.platforms:GAZEBO_SPEC",
            "mujoco = robot_runtime_sim.platforms:MUJOCO_SPEC",
        ],
    },
)
