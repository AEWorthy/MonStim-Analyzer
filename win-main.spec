# -*- mode: python ; coding: utf-8 -*-

# Before you package with Pyinstaller, be sure to do the following:
    #1: delete the config-user.yml file
    #2: update all version numbers (monstim_gui.__init__ and monstim_gui.splash, and numbers below)

# Run the following command to package the application:
# pyinstaller --clean win-main.spec


import os
import sys
import shutil
from PyInstaller.config import CONF


project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from monstim_gui import __version__
EXE_NAME = f'MonStim Analyzer v{__version__}'
DIST_NAME = f'MonStim_Analyzer_v{__version__}-beta'


a = Analysis( # type: ignore
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

pyz = PYZ(a.pure) # type: ignore

exe = EXE( # type: ignore
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    debug=False, # False for release, True for debug
    bootloader_ignore_signals=False,
    upx=True,
    console=False, # False for release, True for debug
    disable_windowed_traceback=True, # True for release, False for debug.
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.png'
)

coll = COLLECT( # type: ignore
    exe,
    a.binaries,
    a.datas,
    upx=True,
    upx_exclude=['PyQt6', 'Qt6Core.dll', 'Qt6Widgets.dll'],
    name=DIST_NAME
)

# Ensure the dist directory exists and copy the readme.md file to it
os.makedirs(CONF['distpath'], exist_ok=True)
shutil.copy2('docs/readme.md', os.path.join(CONF['distpath'], DIST_NAME))
