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
    author_email="support@neuracore.com",
    description="Neuracore Client Library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/neuracoreai/neuracore",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=2.0.0",
        "requests>=2.31.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "requests-oauthlib==2.0.0",
        "pydantic>=2.10",
        "av==14.2.0",
        "aiortc==1.14.0",
        "aiohttp-sse-client==0.2.1",
        "numpy-stl",
        "wget",
        "uvicorn[standard]==0.42.0",
        "fastapi==0.135.2",
        "psutil",
        "typer>=0.20.0",
        "neuracore_types>=6.0.0,<7.0.0",
        "ordered_set",
        "pyzmq==27.1.0",
        "sqlalchemy>=2.0.0",
        "aiosqlite>=0.19.0",
        "aiohttp>=3.9.0",
        "aiofiles>=23.0.0",
        "aiolimiter",
        "pyee==13.0.0",
        "greenlet",
        "filelock>=3.0.0",
    ],
    extras_require={
        "examples": [
            "matplotlib>=3.3.0",
            "mujoco==2.3.7",
            "pyquaternion>=0.9.5",
        ],
        "mjcf": [
            "mujoco>3",
        ],
        "ml": [
            # Pinned at 2.7.1 to match [import], allowing both extras to be
            # installed together. Can be upgraded up to 2.10.0 (inclusive) at
            # the cost of making [ml] and [import] mutually exclusive. Do NOT
            # upgrade to 2.11.0+: it defaults to CUDA 13 which drops V100
            # support, which Neuracore still uses.
            "torch==2.7.1",
            "torchvision==0.22.1",
            "transformers==4.53.2",
            "huggingface-hub==0.36.0",
            "diffusers==0.35.1",
            "safetensors==0.6.2",
            "einops",
            "hydra-core>=1.3.0",
            "tensorboard>=2",
            "names-generator>=0.2.0",
        ],
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "pytest-asyncio>=0.15.1",
            "pytest-xdist",
            "twine>=3.4.2",
            "requests-mock>=1.9.3",
            "pre-commit",
            "types-aiofiles",
        ],
        "import": [
            # lerobot is used only for dataset importing
            # (LeRobotDataset/LeRobotDatasetMetadata).
            # Do NOT install lerobot[transformers-dep] — that extra requires
            # transformers<4.52.0 which conflicts with our transformers==4.53.2 pin.
            # Pinned at 0.3.3 to retain dataset v2 format support;
            # lerobot>=0.4.x drops v2 datasets and requires av>=15.0.0
            # and huggingface-hub<0.36.0.
            # torch is pinned to lerobot 0.3.3's max supported version (2.7.1)
            # and matches [ml], allowing both extras to coexist. Once [ml] is
            # upgraded beyond 2.7.1, this pin will intentionally conflict with
            # [ml], enforcing mutual exclusion. The importer only uses torch for
            # CPU tensor ops (data loading + .numpy()), so CUDA version does
            # not matter.
            "lerobot==0.3.3",
            "torch==2.7.1",
            "huggingface-hub==0.36.0",
            "tensorflow-datasets>=4.9.9",
            "tensorflow>=2.20.0",
            "pin-pink>=4",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuracore = neuracore.core.cli.app:main",
            "nc-data-daemon = neuracore.data_daemon.main:main",
        ]
    },
    keywords="robotics machine-learning ai client-library",
)
