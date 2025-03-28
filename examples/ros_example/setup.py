import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ros_example"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=[
        "setuptools",
        "numpy",
        "opencv-python",
        "neuracore",
    ],
    zip_safe=True,
    author="Robot Learning Team",
    author_email="team@example.com",
    maintainer="Robot Learning Team",
    maintainer_email="team@example.com",
    keywords=["ROS2", "robot learning", "data collection", "simulation"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Topic :: Software Development",
    ],
    description="ROS2 simulation for bimanual robot data collection",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "simulation_node = ros_example.simulation_node:main",
            "data_logger_node = ros_example.data_logger_node:main",
            "action_generator_node = ros_example.action_generator_node:main",
            # 'traditional_sync_node = ros_example.traditional_sync_node:main',
        ],
    },
)
