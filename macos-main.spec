# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
import subprocess

from PyInstaller.config import CONF

main_path = os.path.abspath('main.py')
base_path = os.path.dirname(main_path)

a = Analysis(
    ['main.py'],
    pathex=[base_path, os.path.join(base_path, 'monsit_gui'), os.path.join(base_path, 'monstim_analysis'), os.path.join(base_path, 'monstim_converter')],
    binaries=[],
    datas=[(os.path.join(base_path, 'src', 'icon.png'), 'src'), (os.path.join(base_path, 'src', 'icon.icns'), 'src'), (os.path.join(base_path, 'readme.md'), 'src')],
    hiddenimports=[],
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
    name='macos-monstim-analyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.icns',
)
app = BUNDLE(
    exe,
    name='MonStim Analyzer v1.0.app',
    icon='src/icon.icns',
    bundle_identifier=None,
)

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the additional files to the dist directory
shutil.copy2('config.yml', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))
shutil.copy2('readme.md', os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0'))


# Sign the application with self-signed certificate
subprocess.call([
    'codesign', '--deep', '--force', '--verify', '--verbose',
    '--sign', 'Andrew Worthy',
    os.path.join(CONF['distpath'], 'MonStim Analyzer v1.0.app')
])