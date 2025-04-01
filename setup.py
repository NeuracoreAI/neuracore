from setuptools import find_packages, setup

version = None
with open("neuracore/__init__.py", encoding="utf-8") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.strip().split()[-1][1:-1]
            break
assert version is not None, "Could not find version string"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="neuracore",
    version=version,
    author="Stephen James",
    author_email="stephen@neuraco.com",
    description="Neuracore Client Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuraco/neuracore",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "requests>=2.31.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "requests-oauthlib",
        "pydantic",
        "av",
        "aiortc",
        "aiohttp-sse-client",
        "numpy-stl",
    ],
    extras_require={
        "examples": [
            "matplotlib>=3.3.0",
            "mujoco==2.3.7",
            "dm_control==1.0.14",
            "pyquaternion>=0.9.5",
        ],
        "mjcf": [
            "mujoco>3",
        ],
        "local_endpoint": [
            "torchserve",
            "nvgpu",
            "torch",
            "torchvision",
            "torch-model-archiver",
        ],
        "ml": [
            "torch",
            "torchvision",
        ],
        "upload": [
            "robot_descriptions @ git+https://github.com/stepjam/robot_descriptions.py.git@main",
            "pin",
            "pin-pink",
            "lerobot @ git+https://github.com/huggingface/lerobot.git",
            "tensorflow-datasets",
            "tensorflow",
            "gcsfs",
            "apache_beam",
        ],
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "pytest-asyncio>=0.15.1",
            "twine>=3.4.2",
            "requests-mock>=1.9.3",
            "pre-commit",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuracore-generate-api-key = neuracore.generate_api_key:main",
        ]
    },
    keywords="robotics machine-learning ai client-library",
)
