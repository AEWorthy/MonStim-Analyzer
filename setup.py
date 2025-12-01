import re
from pathlib import Path

from setuptools import find_packages, setup

from monstim_gui.version import VERSION


def load_install_requires(req_path="requirements.txt", exclude=None):
    """Load install requirements from a requirements.txt file while excluding dev-only packages.

    This keeps `setup.py` in sync with `requirements.txt` so Dependabot updates affect packaging.
    """
    if exclude is None:
        exclude = {
            "pytest",
            "setuptools",
            "flake8",
            "black",
            "isort",
            "bandit",
            "safety",
            "pytest-qt",
            "pyqt6",
            "pyqt",
        }

    req_file = Path(req_path)
    if not req_file.exists():
        return []

    installs = []
    for raw in req_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip extras and environment markers (keep simple equality specs)
        # Keep the original spec (e.g., package==1.2.3) unless it's in the exclude list
        name = re.split(r"[=<>!~\[]", line, maxsplit=1)[0].strip().lower()
        if name in exclude:
            continue
        installs.append(line)
    return installs


setup(
    name="MonStim_Analysis",
    version=VERSION,
    description="EMG data analysis and plotting toolkit for MonStim exported CSV data",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="Andrew Worthy",
    author_email="aeworth@emory.edu",
    url="https://github.com/AEWorthy/MonStim_Analysis",
    license="BSD-2-Clause",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=load_install_requires(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
