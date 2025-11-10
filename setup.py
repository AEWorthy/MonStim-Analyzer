from setuptools import find_packages, setup

from monstim_gui.version import VERSION

setup(
    name="MonStim_Analysis",
    version=VERSION,
    description="EMG data analysis and plotting toolkit for MonStim exported CSV data",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Andrew Worthy",
    author_email="aeworth@emory.edu",
    url="https://github.com/AEWorthy/MonStim_Analysis",
    license="BSD-2-Clause",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "Markdown==3.9",
        "PyQt6==6.10.0",
        "PyQt6_sip==13.10.2",
        "PyYAML==6.0.3",
        "h5py==3.15.1",
        "matplotlib==3.10.7",
        "numpy==2.2.6",
        "pandas==2.3.3",
        "packaging==25.0",
        "python_markdown_math==0.9",
        "pytest==9.0.0",
        "scipy==1.15.3",
        "setuptools==80.9.0",
        "traitlets==5.14.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
