from setuptools import setup, find_packages

setup(
    name='mlserve',
    version='0.1',
    description='mlserve is a machine learning serving tool',
    url='http://github.com/gfalcone/mlserve',
    author='gfalcone',
    author_email='gfalcone@github.com',
    license='Apache',
    packages=find_packages(),
    zip_safe=False
)
