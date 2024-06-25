from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="cqt-jax",
    version="0.1.0",
    author="emmanuelinfante",
    author_email="emanuel.06.educacion@gmail.com",
    description="A JAX implementation of Constant-Q Transform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/emmanuelinfante/cqt-jax",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "jax>=0.2.21",
        "jaxlib>=0.1.71",
    ],
)
