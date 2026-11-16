from glob import glob

from setuptools import find_packages, setup

package_name = "agent_eval"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config/tasks", glob("config/tasks/*.yaml")),
        ("share/" + package_name + "/config/recipes", glob("config/recipes/*.yaml")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="leiziting",
    maintainer_email="leiziting@huawei.com",
    description="Platform-neutral Agent evaluation: task specs, the runtime's evaluation extension, "
    "episode runner, suites and run reports",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "run_suite = agent_eval.suite:main",
            "run_episodes = agent_eval.runner_node:main",
        ],
        # Platform plugins register in the "agent_eval.platforms" group from their own packages
        # (agent_eval_robodojo), which may ship suites; agent_eval itself registers none.
    },
)
