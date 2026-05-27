from glob import glob

from setuptools import find_packages, setup

package_name = "sim_models"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        # ament resource index marker
        ("share/ament_index/resource_index/packages", ["resource/sim_models"]),
        # package.xml
        ("share/sim_models", ["package.xml"]),
        # empty scene (static world files, no meshes)
        ("share/sim_models/scenes/empty", glob("scenes/empty/*")),
        # pick_banana scene: layout + templates (NOT meshes — separate entry below)
        (
            "share/sim_models/scenes/pick_banana",
            glob("scenes/pick_banana/*.yaml") + glob("scenes/pick_banana/*.template"),
        ),
        # pick_banana mesh assets (OBJ, STL, PNG textures)
        (
            "share/sim_models/scenes/pick_banana/meshes",
            glob("scenes/pick_banana/meshes/*.obj")
            + glob("scenes/pick_banana/meshes/*.stl")
            + glob("scenes/pick_banana/meshes/*.png"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="zll",
    maintainer_email="zll@example.com",
    description="Scene assets and scene compiler for IB-Robot simulation",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [],
    },
)
