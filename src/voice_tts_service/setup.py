from glob import glob

from setuptools import find_packages, setup

package_name = "voice_tts_service"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["numpy", "setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="liuweihong",
    maintainer_email="liuweihong8@huawei.com",
    description="Manifest-backed typed ZipVoice TTS service for IB-Robot",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "audio_playback_node = voice_tts_service.audio_playback_node:main",
            "package_zipvoice_310p = voice_tts_service.package_zipvoice_310p:main",
        ],
        # Cross-package launch-builder registration (openspec change
        # ``migrate-business-launch-builders``): robot_config discovers this
        # group when building the launch registry, so the voice TTS launch
        # wiring registers from its owning package without any robot_config
        # change. Installing this package does not enable the entry;
        # participation still comes from the robot YAML (``voice_tts.enabled``).
        "ibrobot.launch_builders": [
            "voice_tts = voice_tts_service.launch_builders.voice_tts:VOICE_TTS_BUILDER",
        ],
    },
)
