from glob import glob
import os

from setuptools import setup

package_name = "fusion_core"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml") + glob("config/*.txt"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Fusion Team",
    maintainer_email="team@example.com",
    description="ROS 2 Humble core node for unified sensor fusion.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fusion_node = fusion_core.fusion_node:main",
            "fusion_oosm_regression = fusion_core.oosm_regression:main",
            "synthetic_camera_publisher = fusion_core.synthetic_camera_publisher:main",
            "synthetic_lidar_publisher = fusion_core.synthetic_lidar_publisher:main",
            "synthetic_motion_publisher = fusion_core.synthetic_motion_publisher:main",
            "synthetic_track_publisher = fusion_core.synthetic_track_publisher:main",
            "cone_visualizer_node = fusion_core.cone_visualizer_node:main",
            "interface_conformance_checker = fusion_core.interface_conformance_checker:main",
            "fusion_regression_suite = fusion_core.regression_suite:main",
            "fsd_rosbag_benchmark = fusion_core.fsd_rosbag_benchmark:main",
        ],
    },
)
