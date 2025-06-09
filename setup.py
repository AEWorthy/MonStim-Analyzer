from setuptools import setup, find_packages
from __init__ import __title__ as title
from __init__ import __description__ as description
from __init__ import __version__ as version
from __init__ import __author__ as author
from __init__ import __email__ as author_email

setup(
    name=title,
    version=version,
    description=description,
    author=author,
    author_email=author_email,
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "Markdown==3.6",
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
)
