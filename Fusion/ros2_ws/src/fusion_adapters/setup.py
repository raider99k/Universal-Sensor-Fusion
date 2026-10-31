from glob import glob
import os

from setuptools import setup

package_name = "fusion_adapters"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    package_data={package_name: ["models/*.onnx"]},
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="Fusion Team",
    maintainer_email="team@example.com",
    description="Adapter nodes for ZED/Ouster integration into unified fusion interfaces.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_detection_adapter = fusion_adapters.camera_detection_adapter:main",
            "ouster_lidar_adapter = fusion_adapters.ouster_lidar_adapter:main",
            "onnx_camera_detector = fusion_adapters.onnx_camera_detector:main",
            "lidar_cluster_adapter = fusion_adapters.lidar_cluster_adapter:main",
        ],
    },
)
