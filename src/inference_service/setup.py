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
        "av>=15,<16",
        "rclpy",
        "sensor_msgs",
        "geometry_msgs",
        "diagnostic_msgs",
        "trajectory_msgs",
        "std_msgs",
        "inference_manifest",
        "observation_transport",
        "torch_models",
    ],
    zip_safe=True,
    maintainer="xqw",
    maintainer_email="wuxiaoqiang.rtos@huawei.com",
    description="Multi-model inference service for IB-Robot integration",
    license="Apache-2.0",
    python_requires=">=3.10",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "pipeline_policy_node = inference_service.pipeline_policy_node:main",
            "pure_inference_node = inference_service.pure_inference_node:main",
            "global_inference_scheduler_node = inference_service.global_inference_scheduler_node:main",
            "model_service_node = inference_service.model_service_node:main",
            "recording_node = inference_service.recording_node:main",
        ],
        # Cross-package launch-builder registration (openspec change
        # ``migrate-business-launch-builders``): robot_config discovers this
        # group when building the launch registry, so the model-service
        # nodes' launch wiring registers from its owning package without any
        # robot_config change. Installing this package does not enable the
        # entry; participation still comes from the robot YAML
        # (``perception_services.services``).
        "ibrobot.launch_builders": [
            "perception_models = inference_service.launch_builders.perception_models:PERCEPTION_MODELS_BUILDER",
        ],
    },
)
