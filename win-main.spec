# -*- mode: python ; coding: utf-8 -*-

# Before you package with Pyinstaller, be sure to do the following:
    #1: delete the config-user.yml file
    #2: update all version numbers (.__init__, monstim_gui.__init__ and monstim_gui.splash)

# Run the following command to package the application:
# pyinstaller --clean win-main.spec


import os
import shutil
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files

EXE_NAME = 'MonStim Analyzer v0.2.1'
DIST_NAME = 'MonStim Analyzer v0.2.1-alpha'

# Windows build

a = Analysis(
    ['main.py'],
    pathex=[os.path.dirname(os.path.abspath('main.py'))],
    binaries=[],
    datas=[('src', 'src'), ('docs', 'docs')], # add config.yml and readme.md to datas. Add Settings option to the GUI under File
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
    name=EXE_NAME,
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
    name=DIST_NAME
)

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the example_expts directory to the dist directory
shutil.copytree(os.path.join('files_to_process','example_expts'), os.path.join(CONF['distpath'], DIST_NAME, 'Example Experiments'))
shutil.copy2('docs/readme.md', os.path.join(CONF['distpath'], DIST_NAME))
