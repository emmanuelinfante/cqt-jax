from setuptools import find_packages, setup

setup(
    name="cqt-jax",
    packages=find_packages(exclude=[]),
    version="0.0.1",
    license="MIT",
    description="CQT JAX",
    long_description_content_type="text/markdown",
    author="emmanuelinfante",
    author_email="emanuel.06.educacion@gmail.com",
    url="https://github.com/emmanuelinfante/cqt-jax",
    keywords=["artificial intelligence", "deep learning", "signal processing"],
    install_requires=[
        "jax>=0.2.21",
        "jaxlib>=0.1.71",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
