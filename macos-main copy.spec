# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_all, collect_submodules

# Determine the path to main.py and import_all.py
main_path = os.path.abspath('main.py')
import_all_path = os.path.abspath('import_all.py')
base_path = os.path.dirname(main_path)

# Collect all necessary packages
packages = [
    'numpy',
    'pandas',
    'PyQt6',
    'matplotlib',
    'scipy',
    'seaborn',
    'openpyxl',
    'yaml',
    'monstim_analysis',
    'monstim_gui',
    'monstim_converter'
]

collected_datas = []
collected_binaries = []
collected_hiddenimports = []

for package in packages:
    datas, binaries, hiddenimports = collect_all(package)
    collected_datas.extend(datas)
    collected_binaries.extend(binaries)
    collected_hiddenimports.extend(hiddenimports)

# Collect all submodules of your package
collected_hiddenimports.extend(collect_submodules('monstim_analysis'))
collected_hiddenimports.extend(collect_submodules('monstim_gui'))
collected_hiddenimports.extend(collect_submodules('monstim_converter'))


a = Analysis(
    [main_path],
    pathex=[base_path],
    binaries=collected_binaries,
    datas=collected_datas + [(os.path.join(base_path, 'src', 'icon.png'), 'src')],
    hiddenimports=collected_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False, # change to False for release, True for debug
    optimize=1, # change to 1 for release, 0 for debug
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MonStimAnalyzer-OSX',
    debug=False, # change to False for release, True for debug
    bootloader_ignore_signals=False,
    strip=True, # change to True for release, False for debug
    upx=False,
    console=False, # change to False for release, True for debug
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.png',
)

app = BUNDLE(
    exe,
    name='MonStim Analyzer v1.0',
    icon='src/icon.icns',
    bundle_identifier=None,
)

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the additional files to the dist directory
shutil.copy2('config.yml', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))
shutil.copy2('readme.md', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))
