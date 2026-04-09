from pathlib import Path
from setuptools import setup
from setuptools import find_packages
from glob import glob

package_name = 'robot_config'


def _package_data_files(base_dir: str):
    entries = []
    for path in sorted(Path(base_dir).rglob("*")):
        if not path.is_file():
            continue
        install_dir = f"share/{package_name}/{path.parent.as_posix()}"
        entries.append((install_dir, [str(path)]))
    return entries

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ] + _package_data_files('config'),
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    maintainer='xqw',
    maintainer_email='wuxiaoqiang.rtos@huawei.com',
    description='Unified robot configuration system for ros2_control and peripherals',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'wait_for_clock = robot_config.wait_for_clock:main',
            'wait_for_controllers = robot_config.wait_for_controllers:main',
        ],
    },
)
