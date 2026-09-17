# -*- mode: python ; coding: utf-8 -*-

# Run the following command to package the application:
# pyinstaller --clean win-main.spec


import os
import sys
import shutil
import json
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
DEMO_ARCHIVE = os.path.join(project_root, 'docs', 'resources', 'demo_experiments', 'monstim-synthetic-protocol-demos.zip')

# The synthetic demo archive is a required release resource. Keep it under
# docs/resources so both the frozen application and its bundled help resolve
# one canonical copy; fail packaging rather than silently omitting it.
if not os.path.isfile(DEMO_ARCHIVE):
    raise FileNotFoundError(f'Required bundled demo archive is missing: {DEMO_ARCHIVE}')

datas = []
datas += collect_data_files('assets')
# Preserve the complete nested documentation tree in the bundle: help topics,
# developer references, configuration/profile resources, and the required
# synthetic-demo archive all resolve from it.
datas += collect_data_files('docs')
datas += collect_data_files('numpy')
datas += collect_data_files('scipy')
datas += collect_data_files('matplotlib')
datas += collect_data_files('PySide6')

hiddenimports = ['numpy', 'scipy', 'matplotlib', 'PySide6', 'cryptography.hazmat.primitives.asymmetric.ed25519']

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

# The updater is intentionally a separate, minimal executable. It waits for
# MonStim to exit, selects an already verified staged version, and restarts it;
# it never has access to managed experiment data.
updater_a = Analysis( # type: ignore  # noqa: F821
    ['updater_main.py'],
    pathex=[os.path.dirname(os.path.abspath('updater_main.py'))],
    binaries=[],
    datas=[],
    hiddenimports=['cryptography.hazmat.primitives.asymmetric.ed25519'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
updater_pyz = PYZ(updater_a.pure) # type: ignore
updater_exe = EXE( # type: ignore
    updater_pyz,
    updater_a.scripts,
    updater_a.binaries,
    # Keep the helper build product in PyInstaller's work directory. Its
    # dependencies are collected below into the application's _internal
    # directory, alongside the main application's dependencies.
    exclude_binaries=True,
    name='MonStim Updater',
    debug=False,
    bootloader_ignore_signals=False,
    upx=True,
    console=False,
    disable_windowed_traceback=True,
)

# Do not pass ``updater_exe`` directly to COLLECT: PyInstaller treats every EXE
# argument as a user-facing root-level executable. Collect its executable and
# dependencies as binary entries instead, which places them in ``_internal``.
updater_internal_files = [
    (os.path.basename(updater_exe.name), updater_exe.name, 'BINARY'),
    *updater_exe.dependencies,
]

coll = COLLECT( # type: ignore
    exe,
    a.binaries,
    a.datas,
    updater_internal_files,
    upx=True,
    upx_exclude=['PySide6', 'Qt6Core.dll', 'Qt6Widgets.dll'],
    name=DIST_NAME
)

# Ensure the dist directory exists and copy only the user-facing top-level files.
# The complete user guide is already bundled as ``_internal/docs/user/using_monstim.md``.
os.makedirs(CONF['distpath'], exist_ok=True)
shutil.copy2('QUICKSTART.md', os.path.join(CONF['distpath'], DIST_NAME))
shutil.copy2('LICENSE', os.path.join(CONF['distpath'], DIST_NAME, 'LICENSE.txt'))
shutil.copy2('NOTICE', os.path.join(CONF['distpath'], DIST_NAME, 'NOTICE.txt'))
shutil.copy2('CITATION.cff', os.path.join(CONF['distpath'], DIST_NAME, 'CITATION.cff'))
internal_dir = os.path.join(CONF['distpath'], DIST_NAME, '_internal')
os.makedirs(internal_dir, exist_ok=True)
shutil.copy2('LICENSE', os.path.join(internal_dir, 'LICENSE'))
with open(os.path.join(internal_dir, 'monstim-release.json'), 'w', encoding='utf-8') as f:
    json.dump({'version': VERSION, 'executable': f'{EXE_NAME}.exe'}, f, indent=2)

# PyInstaller 6 builds a standalone EXE into ``dist`` when it is configured as
# a one-file target. Remove the exact legacy path left by earlier specs; the
# helper now builds in ``build`` and is collected only under ``_internal``.
legacy_updater = os.path.join(CONF['distpath'], 'MonStim Updater.exe')
if os.path.isfile(legacy_updater):
    os.remove(legacy_updater)
