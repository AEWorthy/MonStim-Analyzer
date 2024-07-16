# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files

# Windows build

a = Analysis(
    ['main.py'],
    pathex=[os.path.dirname(os.path.abspath('main.py'))],
    binaries=[],
    datas=[('src/icon.png', 'src'), ('readme.md', 'src')], # add config.yml and readme.md to datas. Add Settings option to the GUI under File
    hiddenimports=[],
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
    name='MonStim Analyzer v1.0',
    debug=False, # change to False for release, True for debug
    bootloader_ignore_signals=False,
    strip=True, # change to True for release, False for debug
    upx=True,
    console=False, # change to False for release, True for debug
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.png'
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=True, # change to True for release, False for debug
    upx=True,
    upx_exclude=['PyQt6'],
    name='MonStim Analyzer v1.0'
)

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the additional files to the dist directory
shutil.copy2('config.yml', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))
shutil.copy2('readme.md', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))