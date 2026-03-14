from setuptools import setup, find_packages

setup(
    name="epba-shared",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "pydantic",
        "pyyaml",
        "python-dotenv",
        "structlog"
    ],
)
