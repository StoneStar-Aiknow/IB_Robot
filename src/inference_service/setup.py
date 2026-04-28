import os
from glob import glob

from setuptools import find_packages, setup

package_name = "inference_service"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=[
        "rclpy",
        "sensor_msgs",
        "geometry_msgs",
        "diagnostic_msgs",
        "trajectory_msgs",
        "std_msgs",
    ],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Multi-model inference service for IB-Robot integration",
    license="Apache-2.0",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "lerobot_policy_node = inference_service.lerobot_policy_node:main",
            "pure_inference_node = inference_service.pure_inference_node:main",
            "yolo_graspnet_node = inference_service.yolo_graspnet_node:main",
            "mock_inference_node = inference_service.mock_inference_node:main",
            "simple_mock_inference = inference_service.simple_mock_inference:main",
            "test_system = inference_service.test_system:main",
            "export_onnx_node = inference_service.export_onnx_node:main",
            "loss_compare_node = inference_service.loss_compare_node:main",
        ],
    },
)
