from setuptools import find_packages, setup

package_name = 'model_utils'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'onnx', 'onnxsim', 'torch', 'tqdm'],
    zip_safe=True,
    maintainer='lwh',
    maintainer_email='liuweihong8@huawei.com',
    description='Model utilities for exporting and comparing ONNX models',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
        ],
    },
)
