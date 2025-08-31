# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(
    ['run_video.py'],
    pathex=[],
    binaries=[],
    datas = [
        ("input/*", "input"),
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
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='run_video',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  
)