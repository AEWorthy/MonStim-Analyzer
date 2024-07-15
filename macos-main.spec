# -*- mode: python ; coding: utf-8 -*-

import os
import shutil
from PyInstaller.config import CONF


# Determine the path to main.py and import_all.py
main_path = os.path.abspath('main.py')
base_path = os.path.dirname(main_path)

# Include the Python shared library explicitly
python_lib = '/opt/miniconda3/envs/alv_lab/lib/libpython3.12.dylib'



a = Analysis(
    [main_path],
    pathex=[base_path],
    binaries=[(python_lib, 'libpython3.12.dylib')],
    datas=[(os.path.join(base_path, 'src', 'icon.png'), 'src')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=True, # change to False for release, True for debug
    optimize=0, # change to 1 for release, 0 for debug
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MonStimAnalyzer-OSX',
    debug=True, # change to False for release, True for debug
    bootloader_ignore_signals=False,
    strip=False, # change to True for release, False for debug
    upx=False,
    console=False, # change to False for release, True for debug
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.png'
)

app = BUNDLE(
    exe,
    name='MonStim Analyzer v1-0.app',
    icon='src/icon.icns',
    bundle_identifier=None,
)

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the additional files to the dist directory
shutil.copy2('config.yml', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))
shutil.copy2('readme.md', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))

# To compile, use command: /opt/miniconda3/envs/alv_lab/bin/pyinstaller --clean macos-main.spec