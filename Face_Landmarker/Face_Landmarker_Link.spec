# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

# Define the base path of your script
base_path = Path(__file__).parent.resolve()

# Optional: include face_landmarker.task model file if it's used at runtime
datas = [
    (str(base_path / 'Face_Landmarker' / 'face_landmarker.task'), '.'),
]

block_cipher = None

a = Analysis(
    ['Face_Landmarker/Face_Landmarker_Link.py'],  # adjust path if needed
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
    console=True,  # set to False if it's a GUI app
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
