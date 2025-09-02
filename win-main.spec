# -*- mode: python ; coding: utf-8 -*-

# Run the following command to package the application:
# pyinstaller --clean win-main.spec


import os
import sys
import shutil
from PyInstaller.config import CONF

# Set project root
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Delete 'config-user.yml' in ./docs/
shutil.rmtree(os.path.join(project_root, 'docs', 'config-user.yml'), ignore_errors=True)

# Set dist name with version
from monstim_gui.version import VERSION
EXE_NAME = f'MonStim Analyzer v{VERSION}'
DIST_NAME = f'MonStim_Analyzer_v{VERSION}-WIN'


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
    icon='src/icon.ico'
)

coll = COLLECT( # type: ignore
    exe,
    a.binaries,
    a.datas,
    upx=True,
    upx_exclude=['PyQt6', 'Qt6Core.dll', 'Qt6Widgets.dll'],
    name=DIST_NAME
)

# Ensure the dist directory exists, and copy the readme.md and QUICKSTART.md files to it
os.makedirs(CONF['distpath'], exist_ok=True)
shutil.copy2('docs/readme.md', os.path.join(CONF['distpath'], DIST_NAME))
shutil.copy2('QUICKSTART.md', os.path.join(CONF['distpath'], DIST_NAME))

