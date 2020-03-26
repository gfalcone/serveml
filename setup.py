from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="serveml",
    version="0.2.1",
    author="gfalcone",
    author_email="gfalcone@github.com",
    description="serveml is a machine learning serving tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    url="http://github.com/gfalcone/mlserve",
    packages=["serveml"],
    license="Apache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    zip_safe=False,
)
