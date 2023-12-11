from setuptools import setup, find_packages

setup(
    name='cranpose',
    version='0.0.1',
    install_requires=[
        'numpy>=1.25.2',
        'opencv-contrib-python==4.8.1.78',
        'scipy>=1.11.2',
        'jupyter>=1.0.0',
        'matplotlib>=3.7.3',
        'pykalman @ git+https://github.com/pykalman/pykalman@8d3f8e4',
    ],
    author='Viacheslav Martynov, NKBTech',
    packages=find_packages(
        where='cranpose',
    )
)
