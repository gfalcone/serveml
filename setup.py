from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="mlserve",
    version="0.1",
    author="gfalcone",
    author_email="gfalcone@github.com",
    description="mlserve is a machine learning serving tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://github.com/gfalcone/mlserve",
    packages=find_packages(),
    license="Apache",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=False,
)
