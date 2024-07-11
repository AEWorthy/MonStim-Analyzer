# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_submodules

hidden_imports = collect_submodules('gui')
project_dir = r'C:\\Users\\aewor\\Documents\\GitHub\\MonStim_Analysis\\EMGAnalysis.spec'


a = Analysis(
    ['main.py'],
    pathex=[project_dir, os.path.dirname(os.path.abspath('main.py'))],
    binaries=[],
    datas=[],
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MonStim Analyzer v1.0-debugger',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.png'
)

import shutil
from PyInstaller.config import CONF

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the additional files to the dist directory
shutil.copy2('config.yml', CONF['distpath'])
shutil.copy2('readme.txt', CONF['distpath'])