from glob import glob

from setuptools import find_packages, setup

package_name = "robot_calibration"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/examples", glob("config/examples/*.yaml")),
        ("share/" + package_name + "/config/fast_calib/scenes", glob("config/fast_calib/scenes/*.yaml")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools", "numpy", "pyyaml"],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Sensor calibration capture, artifact, validation, and activation workflows",
    license="Apache-2.0",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "calib_check = robot_calibration.sensor_calibration:main",
            "calib_capture_finalize = robot_calibration.capture:finalize_cli",
            "calib_offline = robot_calibration.cli:main",
            "calib_capture = robot_calibration.workflow:capture_main",
            "calib_capture_preview = robot_calibration.capture_preview:main",
            "calib_preview_decode = robot_calibration.preview_decode:main",
            "calib_process = robot_calibration.workflow:solve_main",
            "calib_validate = robot_calibration.validation:main",
            "calib_view = robot_calibration.viewer:main",
            "calib_overlay = robot_calibration.live_overlay:main",
            "calib_approve = robot_calibration.approval:main",
        ],
    },
)
