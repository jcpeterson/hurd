from codecs import open
from os import path

from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(here, "requirements.txt"), encoding="utf-8") as f:
    requirements = f.read().split("\n")

setup(
    name="hurd",
    version="0.0.1",
    author="Joshua Peterson and David Bourgin",
    description="Computational models of human decision making",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcpeterson/hurd",
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    project_urls={"Source": "https://github.com/jcpeterson/hurd"},
)
