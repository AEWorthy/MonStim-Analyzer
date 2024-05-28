
Things to do before publishing:
    - Make package _init_ file for whole package.
    - setup.py
    - requirements.txt
    - README.md
    - MANIFEST.in



EX. Setup:

from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        # List your dependencies here
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A brief description of your package',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_package',  # Replace with your repo URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)