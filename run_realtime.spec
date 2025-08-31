# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['run_realtime.py'],
    pathex=[],
    binaries=[],
    # Add model files
    datas = [
        ("input/THE GRAND BOUDABEST HOTEL The Police Are Here .mp4", "input"),
        ("config/config.yaml", "config"),
        # FFHQ rebalanced
        ('models/ffhqrebalanced512-128.pkl', 'models'),

        # YOLO face model
        ('models/yolov8n-face.pt', 'models'),

        # DiscoFaceGAN checkpoint files
        ('models/models_discofacegan/stage1_epoch_395.ckpt.data-00000-of-00001', 'models/models_discofacegan'),
        ('models/models_discofacegan/stage1_epoch_395.ckpt.index', 'models/models_discofacegan'),
        ('models/models_discofacegan/stage1_epoch_395.ckpt.meta', 'models/models_discofacegan'),

        # BeautyGAN checkpoint files
        ('models/models_beautygan/checkpoint', 'models/models_beautygan'),
        ('models/models_beautygan/model.data-00000-of-00001', 'models/models_beautygan'),
        ('models/models_beautygan/model.index', 'models/models_beautygan'),
        ('models/models_beautygan/model.meta', 'models/models_beautygan'),

        # StarGAN checkpoint
        ('StarGAN/stargan_celeba_128/models/200000-G.ckpt', 'StarGAN/stargan_celeba_128/models'),
    ],
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
    name='run_realtime',
    debug=False,
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
)
