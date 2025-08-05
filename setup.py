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
        "Markdown==3.8.2",
        "PyQt6==6.9.1",
        "PyQt6_sip==13.8.0",
        "PyYAML==6.0.2",
        "h5py==3.12.1",
        "matplotlib==3.10.3",
        "numpy==2.2.6",
        "pandas==2.3.0",
        "packaging==25.0",
        "python_markdown_math==0.8",
        "pytest==8.4.0",
        "scipy==1.15.3",
        "setuptools==72.2.0",
        "traitlets==5.14.3",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
