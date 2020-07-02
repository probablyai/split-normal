import runpy
import setuptools


__version__ = runpy.run_path("split_normal/version.py")["__version__"]


def read_readme(filename="README.md"):
    with open(filename, "r") as f:
        readme = f.read()
    return readme


if __name__ == '__main__':
    setuptools.setup(
        name="split-normal",
        version=__version__,
        author="Tárik S. Salem",
        description="A tiny package implementing functions of the split normal distribution compatible with Numpy and JAX.",
        long_description=read_readme(),
        long_description_content_type="text/markdown",
        url="https://github.com/probablyai/split-normal",
        packages=setuptools.find_packages(exclude=("tests", )),
        install_requires=["numpy>=1.17.3", "scipy>=1.4.0", "jax>=0.1.59", "jaxlib>=0.1.40"],
        extras_require={
            "dev": ["pytest>=5.4.2", "readme2tex>=0.0.1b2", "GitPython>=3.1.3"]
        },
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Topic :: Scientific/Engineering",
            "Topic :: Software Development"
        ],
        python_requires=">=3.6",
    )
