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
    author_email="stephen@neuracore.com",
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
        "requests-oauthlib",
        "pydantic>=2.10",
        "av==14.2.0",
        "aiortc",
        "aiohttp-sse-client",
        "numpy-stl",
        "wget",
        "uvicorn[standard]",
        "fastapi",
        "psutil",
        "typer>=0.20.0",
        "neuracore-types>=4.3.0,<5.0.0",
        "ordered_set",
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
            "torch",
            "torchvision",
            "transformers==4.57.3",
            "huggingface-hub==0.36.0",
            "diffusers==0.35.1",
            "safetensors==0.6.2",
            "einops",
            "hydra-core>=1.3.0",
            "tensorboard>=2",
        ],
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.1",
            "pytest-asyncio>=0.15.1",
            "pytest-xdist",
            "twine>=3.4.2",
            "requests-mock>=1.9.3",
            "pre-commit",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuracore = neuracore.core.cli.app:main",
            "nc-login = neuracore.core.cli.generate_api_key:main",
            "nc-select-org = neuracore.core.cli.select_current_org:main",
            "nc-launch-server = neuracore.core.cli.launch_server:main",
        ]
    },
    keywords="robotics machine-learning ai client-library",
)
