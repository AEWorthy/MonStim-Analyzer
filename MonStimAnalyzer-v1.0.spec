# -*- mode: python ; coding: utf-8 -*-
import os
import shutil
from PyInstaller.config import CONF

a = Analysis(
    ['main.py'],
    pathex=[os.path.dirname(os.path.abspath('main.py')), 'src'],
    binaries=[],
    datas=[('src/icon.png', 'src')],
    hiddenimports=['pyi_splash'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

splash = Splash('src\\splash.png',
                binaries=a.binaries,
                datas=a.datas,
                text_pos=None,
                text_size=12,
                minify_script=True)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name='MonStim Analyzer v1.0',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='src/icon.png'
)

# Ensure the dist directory exists
os.makedirs(CONF['distpath'], exist_ok=True)

# Copy the additional files to the dist directory
shutil.copy2('config.yml', CONF['distpath'])
shutil.copy2('readme.md', CONF['distpath'])
