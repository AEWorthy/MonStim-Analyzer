# import_all.py
import importlib
import pkgutil
import sys

def import_and_print(package_name):
    try:
        package = importlib.import_module(package_name)
        print(f"Successfully imported {package_name}")
        print(f"{package_name} path: {package.__file__}")
        return package
    except ImportError as e:
        print(f"Failed to import {package_name}: {e}")
        return None

def import_all_modules(package_name):
    package = import_and_print(package_name)
    if package:
        for _, module_name, _ in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
            import_and_print(module_name)

# Import all modules from your package
import_all_modules('monstim_analysis')
import_all_modules('monstim_gui')
import_all_modules('monstim_converter')

# Explicitly import commonly used packages
packages_to_import = [
    'numpy',
    'pandas',
    'PyQt6',
    'matplotlib',
    'scipy',
    'yaml'
]

for package in packages_to_import:
    import_and_print(package)

print("Python path:")
for path in sys.path:
    print(path)
