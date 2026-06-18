from setuptools import find_packages, setup

package_name = "detection_service"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    package_data={
        "detection_service": ["config/gdino/*.py"],
    },
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="IB-Robot Contributors",
    maintainer_email="roboguru.92@gmail.com",
    description="Detection service: Grounding-DINO + SAM2 for IB-Robot",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "grounded_sam2_node = detection_service.grounded_sam2_node:main",
            "grounded_sam2_snapshot = detection_service.grounded_sam2_snapshot:main",
        ],
    },
)
