from setuptools import setup, find_packages

setup(
    name="cqt-jax",
    version="0.1.0",
    packages=find_packages(exclude=[]),
    author="emmanuelinfante",
    author_email="emanuel.06.educacion@gmail.com",
    license="MIT",
    description="A JAX implementation of Constant-Q Transform",
    url="https://github.com/emmanuelinfante/cqt-jax",
    install_requires=[
        "jax>=0.2.21",
        "jaxlib>=0.1.71",
    ],
)
