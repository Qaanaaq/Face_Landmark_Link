# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# Hardcode the correct path relative to the repo root or use "." if already in that directory
base_path = Path(".").resolve()

datas = [
    (str(base_path / 'Face_Landmarker' / 'face_landmarker.task'), '.'),
]

block_cipher = None

a = Analysis(
    ['Face_Landmarker/Face_Landmarker_Link.py'],
    pathex=[str(base_path)],
    binaries=[],
    datas=datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Face_Landmarker_Link',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Face_Landmarker_Link',
)
