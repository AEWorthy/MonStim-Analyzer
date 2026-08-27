# -*- mode: python ; coding: utf-8 -*-

# Run the following command to package the application:
# pyinstaller --clean win-main.spec


import os
import sys
import shutil
from PyInstaller.config import CONF
from PyInstaller.utils.hooks import collect_data_files

# Set project root
project_root = os.path.abspath(os.getcwd())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Set dist name with version
from monstim_gui.version import VERSION  # noqa: E402
EXE_NAME = f'MonStim Analyzer v{VERSION}'
DIST_NAME = f'MonStim_Analyzer_v{VERSION}-WIN'

datas = []
datas += collect_data_files('assets')
# Preserve the complete nested documentation tree in the bundle: help topics,
# developer references, and configuration/profile resources all resolve from it.
datas += collect_data_files('docs')
datas += collect_data_files('numpy')
datas += collect_data_files('scipy')
datas += collect_data_files('matplotlib')
datas += collect_data_files('PySide6')

hiddenimports = ['numpy', 'scipy', 'matplotlib', 'PySide6']

a = Analysis( # type: ignore  # noqa: F821
    ['main.py'],
    pathex=[os.path.dirname(os.path.abspath('main.py'))],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False, # change to False for release, True for debug
    optimize=1, # change to 1 for release, 0 for debug
)

# A user override must not ship with the application. The default configuration
# and bundled profiles remain under docs/resources. ``Analysis.datas`` differs
# slightly across PyInstaller releases, so inspect both name/path fields.
def is_user_config_data(entry):
    return isinstance(entry, tuple) and any(
        str(field).replace('\\', '/').endswith('docs/resources/config-user.yml')
        for field in entry[:2]
    )


a.datas = [entry for entry in a.datas if not is_user_config_data(entry)]

pyz = PYZ(a.pure) # type: ignore

exe = EXE( # type: ignore
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=EXE_NAME,
    debug=False, # False for release, True for debug
    bootloader_ignore_signals=False, # False for release, True for debug
    upx=True,
    console=False, # False for release, True for debug
    disable_windowed_traceback=True, # True for release, False for debug
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/icon.ico'
)

coll = COLLECT( # type: ignore
    exe,
    a.binaries,
    a.datas,
    upx=True,
    upx_exclude=['PySide6', 'Qt6Core.dll', 'Qt6Widgets.dll'],
    name=DIST_NAME
)

# Ensure the dist directory exists, and copy the user guide and quick-start file to it.
os.makedirs(CONF['distpath'], exist_ok=True)
shutil.copy2('docs/user/using_monstim.md', os.path.join(CONF['distpath'], DIST_NAME))
shutil.copy2('QUICKSTART.md', os.path.join(CONF['distpath'], DIST_NAME))
