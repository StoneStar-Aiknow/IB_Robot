from setuptools import find_packages, setup

package_name = "model_utils"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=[
        "setuptools",
        "inference_manifest",
        "onnx",
        "onnxsim",
        "torch",
        "torchvision",
        "tqdm",
        "Pillow",
        "PyYAML",
    ],
    zip_safe=True,
    maintainer="lwh",
    maintainer_email="liuweihong8@huawei.com",
    description="Model utilities for exporting and comparing ONNX models",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "frame_inspect = model_utils.frame_inspect:main",
            "package-compiled-deployment = model_utils.package_compiled_deployment:main",
            "package-hmm-deployment = model_utils.hmm_export:main",
            "package-torch-deployment = model_utils.package_torch_deployment:main",
            "pi05-export = model_utils.pi05_export.__main__:console_main",
        ],
    },
)
