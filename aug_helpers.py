import os
import glob
import cv2
import numpy as np
import csv
from typing import Dict, List, Optional, Tuple, Any
from io import BytesIO
import yaml
from build_exe import resource_path
# stargan_wrapper.py
import torch
from torchvision import transforms
from PIL import Image
import tensorflow as tf
# eg3d_wrapper.py
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import eg3d.eg3d.legacy as legacy # from eg3d repo
import eg3d.eg3d.dnnlib as dnnlib

from StarGAN.model import Generator  # depends on how StarGAN repo is structured

# ============================================================
# aug_helpers.py — Face augmentations + (optional) GAN hooks
# ============================================================
# What you get:
#   • Geometric + photometric aug (lighting, contrast, noise, vignette, etc.)
#   • Pose/profile jitter (classical + light 3D-ish skew)
#   • Expression, hair, makeup, accessories:
#       - If the referenced GANs are available (see loaders), we call them
#       - Otherwise we fall back to classical/overlay approximations
#   • Realism enhancement pass at the end (dual-path GAN hook or lite enhancer)
#   • A single orchestrator: make_augmented_faces_full(...)
#
# How to plug GANs:
#   GANs mentioned in the article (examples):
#     - Hair: DiscoGAN / StarGAN (multi-domain)  -> hook: 'stargan' or 'discogan'
#     - Makeup: BeautyGAN                        -> hook: 'beautygan'
#     - Accessories (glasses on/off): InfoGAN    -> hook: 'infogan'
#     - Pose/profile: TP-GAN / FF-GAN / X2Face   -> hook: 'pose_gan' (one of these)
#     - Expressions: ExpreGAN                    -> hook: 'expregan'
#     - Generic cGAN for DA: DAGAN               -> hook: 'dagan'
#     - Realism enhancer (dual-path GAN)         -> hook: 'realism_gan'
#
#   You can register any callable that maps BGR np.ndarray -> BGR np.ndarray.
#   See build_gan_registry() for the expected interface.
#
#   This file is import-only; we avoid downloading weights.
#   Provide initialized callables in a registry dict, or paths in config
#   and implement the loader stubs to return your callables.
# ============================================================

# ------------------------
# Basic file helpers
# ------------------------

config_path = resource_path("config/config.yaml")


with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

beautygan_path = resource_path(cfg["paths"]["beautygan_path"])
discofacegan_path = cfg["paths"]["discofacegan_path"]
eg3d_path = resource_path(cfg["paths"]["eg3d_model"])
stargan_path = resource_path(cfg["paths"]["stargan_path"])

device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu') 

def find_one_image_named_1(folder: str) -> Optional[str]:
    """
    Finds a single image whose basename starts with '1' in the given folder.
    Supports common extensions. Returns first match or None.
    """
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    candidates: List[str] = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(folder, f"1{ext[1:]}")))
        candidates.extend(glob.glob(os.path.join(folder, "1.*")))
    if not candidates:
        for ext in exts:
            candidates.extend(glob.glob(os.path.join(folder, f"1*{ext[1:]}")))
    # Dedup & first
    seen = set()
    uniq = []
    for c in candidates:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    return uniq[0] if uniq else None

# --------------------------------
# Photometric helpers (uint8 BGR)
# --------------------------------

def gamma_adjust_uint8(img_bgr: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Fast LUT-based gamma on uint8 BGR image. gamma<1 brightens, >1 darkens."""
    g = max(gamma, 1e-6)
    invGamma = 1.0 / g
    table = (np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)])
             .astype("uint8"))
    return cv2.LUT(img_bgr, table)


def adjust_brightness_contrast(img_bgr: np.ndarray,
                               brightness: float = 0.0,
                               contrast: float = 0.0) -> np.ndarray:
    """Apply brightness/contrast: out = img*alpha + beta, where
    contrast in [-1,1] -> alpha; brightness in [-1,1] -> beta.
    """
    alpha = 1.0 + float(contrast)
    beta = 255.0 * float(brightness)
    out = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)
    return out


def color_jitter_hsv(img_bgr: np.ndarray,
                     hue_shift: int = 0,
                     sat_scale: float = 1.0,
                     val_scale: float = 1.0) -> np.ndarray:
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    if hue_shift != 0:
        h = (h.astype(np.int32) + hue_shift) % 180
        h = h.astype(np.uint8)
    s = np.clip(s.astype(np.float32) * float(sat_scale), 0, 255).astype(np.uint8)
    v = np.clip(v.astype(np.float32) * float(val_scale), 0, 255).astype(np.uint8)
    img_hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)


def add_gaussian_noise(img_bgr: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    noise = np.random.normal(0, sigma, img_bgr.shape).astype(np.float32)
    out = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def add_salt_pepper(img_bgr: np.ndarray, amount: float = 0.002, s_vs_p: float = 0.5) -> np.ndarray:
    out = img_bgr.copy()
    num = int(amount * out.shape[0] * out.shape[1])
    # salt
    coords = (np.random.randint(0, out.shape[0], num), np.random.randint(0, out.shape[1], num))
    out[coords] = 255
    # pepper
    coords = (np.random.randint(0, out.shape[0], num), np.random.randint(0, out.shape[1], num))
    out[coords] = 0
    return out


def vignette(img_bgr: np.ndarray, strength: float = 0.5) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    y, x = np.ogrid[:h, :w]
    cy, cx = h / 2, w / 2
    mask = ((x - cx) ** 2 + (y - cy) ** 2) / (cx ** 2 + cy ** 2)
    mask = np.clip(1.0 - strength * mask, 0, 1)
    out = (img_bgr.astype(np.float32) * mask[..., None]).astype(np.uint8)
    return out

# --------------------------------
# Geometric helpers
# --------------------------------

def random_rotate(img_bgr: np.ndarray, deg: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def random_translate(img_bgr: np.ndarray, tx: int, ty: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT_101)


def random_scale(img_bgr: np.ndarray, sx: float, sy: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), 0, sx)
    scaled = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101)
    # crude anisotropic control via resize if sy!=sx
    if abs(sy - sx) > 1e-3:
        scaled = cv2.resize(scaled, None, fx=1.0, fy=sy/sx, interpolation=cv2.INTER_LINEAR)
        scaled = cv2.resize(scaled, (w, h), interpolation=cv2.INTER_LINEAR)
    return scaled


def horizontal_flip(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.flip(img_bgr, 1)


def yaw_warp(img_bgr: np.ndarray, direction: str = 'left', strength: float = 0.22) -> np.ndarray:
    """Simulate yaw via perspective skew (profile-ish)."""
    h, w = img_bgr.shape[:2]
    s = float(np.clip(strength, 0.0, 0.45))
    src = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    if direction == 'left':
        dst = np.float32([[int(w*s), 0], [w-1, 0], [w-1, h-1], [int(w*s), h-1]])
    else:
        dst = np.float32([[0, 0], [int(w*(1-s))-1, 0], [int(w*(1-s))-1, h-1], [0, h-1]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)

# --------------------------------
# Accessory overlays (fallback when GAN not present)
# --------------------------------

def overlay_png_rgba(base_bgr: np.ndarray, overlay_rgba: np.ndarray,
                     bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Overlay RGBA image into base within bbox=(x1,y1,x2,y2)."""
    x1, y1, x2, y2 = bbox
    out = base_bgr.copy()
    oh, ow = overlay_rgba.shape[:2]
    W = max(1, x2 - x1)
    H = max(1, y2 - y1)
    overlay_resized = cv2.resize(overlay_rgba, (W, H), interpolation=cv2.INTER_AREA)
    bgr = overlay_resized[..., :3]
    alpha = overlay_resized[..., 3:4].astype(np.float32) / 255.0
    roi = out[y1:y2, x1:x2].astype(np.float32)
    blended = roi * (1 - alpha) + bgr.astype(np.float32) * alpha
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    return out

# --------------------------------
# GAN registry & loaders (stubs — plug your own)
# --------------------------------

class GanCallable:
    """Light wrapper: any callable(img_bgr, **kwargs) -> img_bgr."""
    def __init__(self, fn):
        self.fn = fn
    def __call__(self, img_bgr: np.ndarray, **kwargs) -> np.ndarray:
        return self.fn(img_bgr, **kwargs)



# Load generator
def load_stargan(model_path=stargan_path, device="cuda"):
    G = Generator(conv_dim=64, c_dim=5, repeat_num=6)  # params from config.py in repo
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()
    return G.to(device)

# Generate augmented face
def stargan_augment(image_pil, G, device="cuda"):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    x = transform(image_pil).unsqueeze(0).to(device)

    # Example: change attribute (e.g., hair color, gender, etc.)
    c_trg = torch.tensor([[1, 0, 0, 0, 0]]).to(device).float()  # one-hot for 5 attrs
    with torch.no_grad():
        out = G(x, c_trg)

    out = (out.squeeze().cpu() + 1) / 2  # [-1,1] → [0,1]
    out_pil = transforms.ToPILImage()(out.clamp(0,1))
    return out_pil

G = load_stargan(device=device)

def stargan_callable(img_bgr: np.ndarray, ref=None, domain=None) -> np.ndarray:
    # Convert BGR → PIL
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)

    # Run augmentation
    out_pil = stargan_augment(pil, G, device=device)

    # Convert back to BGR np.ndarray
    out_rgb = np.array(out_pil)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr

class BeautyGANWrapper:
    def __init__(self, model_dir=beautygan_path):
        # Disable eager execution (TF2 runs eagerly by default)
        tf.compat.v1.disable_eager_execution()

        # Reset graph
        tf.compat.v1.reset_default_graph()

        # Start TF1-style session
        self.sess = tf.compat.v1.Session()

        # Load graph
        saver = tf.compat.v1.train.import_meta_graph(resource_path(os.path.join(model_dir, "model.meta")))
        saver.restore(self.sess, tf.train.latest_checkpoint(model_dir))

        # Get graph reference
        graph = tf.compat.v1.get_default_graph()

        # Tensors by name
        self.X = graph.get_tensor_by_name("X:0")       # input: no-makeup face
        self.Y = graph.get_tensor_by_name("Y:0")       # input: ref makeup face
        self.Xs = graph.get_tensor_by_name("generator/xs:0")  # output: generated face

    def __call__(self, no_makeup_bgr, ref_makeup_bgr):
        size = 256
        nm = cv2.resize(no_makeup_bgr, (size, size))
        ref = cv2.resize(ref_makeup_bgr, (size, size))

        X_img = ((nm / 255.0 - 0.5) * 2)[None, ...]
        Y_img = ((ref / 255.0 - 0.5) * 2)[None, ...]

        out = self.sess.run(self.Xs, feed_dict={self.X: X_img, self.Y: Y_img})
        res = ((out[0] + 1) / 2 * 255).astype(np.uint8)
        return cv2.cvtColor(res, cv2.COLOR_RGB2BGR)

# ✅ Load BeautyGAN once
beautygan = BeautyGANWrapper(beautygan_path)

def load_eg3d_cpu(pkl_path):
    # read pickle bytes
    with open(pkl_path, 'rb') as f:
        raw_bytes = f.read()

    f_cpu = BytesIO(raw_bytes)

    # Monkey-patch torch.load temporarily to force CPU
    orig_torch_load = torch.load
    torch.load = lambda f, *a, **kw: orig_torch_load(f, map_location='cpu', **kw)
    try:
        data = legacy.load_network_pkl(f_cpu)
    finally:
        torch.load = orig_torch_load  # restore original

    G = data['G_ema'].cpu()
    return G

def eg3d_pose_change(img_bgr: np.ndarray, G, yaw: float = 0.0, device="cuda") -> np.ndarray:
    # Convert BGR → tensor latent input (EG3D expects z, not raw images!)
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to(device)
    c = torch.zeros([1, G.c_dim], device=device)
    
    # yaw in radians; scale factor controls how far side profile goes
    camera_params = torch.tensor([[yaw, 0, 0, 1.0]], device=device)  # [yaw, pitch, roll, dist]
    
    with torch.no_grad():
        img = G(z, c, camera_params, noise_mode='const')['image']
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img[0].cpu().numpy()[..., ::-1]  # to BGR
    return img


EG3D_G = load_eg3d_cpu(eg3d_path)

def eg3d_callable(img_bgr: np.ndarray, driver=None, yaw=None):
    return eg3d_pose_change(img_bgr, EG3D_G, yaw=float(yaw or 0.0), device=device)

class DiscoFaceGANWrapper:
    def __init__(self, model_dir=discofacegan_path):
        # Make sure we're in TF v1 compatibility mode
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.reset_default_graph()
        self.sess = tf.compat.v1.Session()

        # Load the model
        meta_file = resource_path(os.path.join(model_dir, "stage1_epoch_395.ckpt.meta"))
        ckpt_file = resource_path(os.path.join(model_dir, "stage1_epoch_395.ckpt"))  # no extension
        saver = tf.compat.v1.train.import_meta_graph(meta_file)
        saver.restore(self.sess, ckpt_file)

        # Get the graph
        self.graph = tf.compat.v1.get_default_graph()

        # Correct tensor handles
        self.input_image = self.graph.get_tensor_by_name("x:0")
        self.output = self.graph.get_tensor_by_name("rot/stage1/decoder/x_hat/BiasAdd:0")

    def __call__(self, img_bgr):
        # Preprocess input
        img = cv2.resize(img_bgr, (256, 256)) / 255.0
        img = np.expand_dims(img, axis=0)

        # Run model
        feed_dict = {
            self.input_image: img
        }
        out = self.sess.run(self.output, feed_dict=feed_dict)

        # Postprocess output
        return (out[0] * 255).astype(np.uint8)


# Usage
discogan = DiscoFaceGANWrapper(discofacegan_path)

def build_gan_registry(config: Optional[Dict[str, Any]] = None) -> Dict[str, Optional[GanCallable]]:
    """
    Create a dict of optional GAN callables. Provide pre-initialized
    callables in config under keys below OR implement the TODO loader logic
    to create callables from given paths.

    Expected keys:
      'stargan'   : hair/appearance transfer (multi-domain)
      'discogan'  : hair transfer (cross-domain)
      'beautygan' : makeup transfer
      'infogan'   : accessories (e.g., glasses on/off)
      'pose_gan'  : TP-GAN / FF-GAN / X2Face for pose/profile
      'expregan'  : expression editing
      'dagan'     : general DA cGAN
      'realism_gan': dual-path GAN enhancer
      'eg3d_gan': for side profiles
    """
    cfg = config or {}
    reg: Dict[str, Optional[GanCallable]] = {}

    def _from_config(name: str) -> Optional[GanCallable]:
        obj = cfg.get(name)
        if callable(obj):
            return GanCallable(obj)
        # TODO: if obj is a dict with 'weights'/'type', you can instantiate here
        return None

    for k in [
        'stargan', 'discogan', 'beautygan', 'infogan',
        'pose_gan', 'expregan', 'dagan', 'realism_gan', 'eg3d_gan'
    ]:
        reg[k] = _from_config(k)

    return reg


# Example init
gan_registry = build_gan_registry({
    "stargan": stargan_callable,
    "beautygan": beautygan,
    'eg3d_gan': eg3d_callable,
    "discogan": discogan
})
# --------------------------------
# Realism enhancer (lite fallback)
# --------------------------------

def realism_enhance_lite(img_bgr: np.ndarray) -> np.ndarray:
    """A lightweight enhancer mimicking a GAN post-process: unsharp mask + mild grain
    + jpeg-like compression to reduce synthetic artifacts.
    """
    # Unsharp mask
    blur = cv2.GaussianBlur(img_bgr, (0, 0), 1.2)
    sharp = cv2.addWeighted(img_bgr, 1.3, blur, -0.3, 0)
    # Mild film grain
    grain = np.random.normal(0, 3.5, img_bgr.shape).astype(np.float32)
    grainy = np.clip(sharp.astype(np.float32) + grain, 0, 255).astype(np.uint8)
    # JPEG round-trip
    enc = cv2.imencode('.jpg', grainy, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1]
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec

# --------------------------------
# High-level atomic transforms
# --------------------------------

def geometric_photometric_bundle(img_bgr: np.ndarray, seed: Optional[int] = None) -> np.ndarray:
    """One random sample of combined geometric+photometric transforms."""
    rng = np.random.default_rng(seed)
    out = img_bgr.copy()

    # Geometric
    deg = float(rng.uniform(-18, 18))
    tx = int(rng.integers(-0.06 * out.shape[1], 0.06 * out.shape[1]))
    ty = int(rng.integers(-0.06 * out.shape[0], 0.06 * out.shape[0]))
    sx = float(rng.uniform(0.9, 1.1))
    sy = float(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        out = horizontal_flip(out)
    out = random_rotate(out, deg)
    out = random_translate(out, tx, ty)
    out = random_scale(out, sx, sy)

    # Photometric
    if rng.random() < 0.7:
        # brightness [-0.12, 0.12], contrast [-0.25, 0.25]
        out = adjust_brightness_contrast(out, float(rng.uniform(-0.12, 0.12)), float(rng.uniform(-0.25, 0.25)))
    if rng.random() < 0.6:
        out = gamma_adjust_uint8(out, float(rng.uniform(0.7, 1.4)))
    if rng.random() < 0.5:
        out = color_jitter_hsv(out, int(rng.integers(-10, 10)), float(rng.uniform(0.85, 1.15)), float(rng.uniform(0.85, 1.15)))
    if rng.random() < 0.35:
        out = add_gaussian_noise(out, float(rng.uniform(3, 10)))
    if rng.random() < 0.15:
        out = add_salt_pepper(out, amount=float(rng.uniform(0.001, 0.004)))
    if rng.random() < 0.3:
        out = vignette(out, float(rng.uniform(0.15, 0.45)))

    return out


def pose_profile_variants(img_bgr: np.ndarray) -> List[np.ndarray]:
    """Simple classical pose/profile jitter: left/right yaw."""
    return [yaw_warp(img_bgr, 'left', 0.18), yaw_warp(img_bgr, 'right', 0.22)]

# --------------------------------
# GAN-driven domain edits (with fallbacks)
# --------------------------------

def apply_hair_change(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]],
                      ref_img_bgr: Optional[np.ndarray] = None) -> np.ndarray:
    if reg.get('stargan') is not None:
        return reg['stargan'](img_bgr, ref=ref_img_bgr, domain='hair')
    if reg.get('discogan') is not None:
        return reg['discogan'](img_bgr, ref=ref_img_bgr)
    # Fallback: slight color-shift around hair-like regions using a crude top-band heuristic
    h, w = img_bgr.shape[:2]
    band = img_bgr[0:int(h*0.28), :]
    band = color_jitter_hsv(band, hue_shift=np.random.randint(-12, 12), sat_scale=np.random.uniform(0.9, 1.2))
    out = img_bgr.copy()
    out[0:int(h*0.28), :] = band
    return out


def apply_makeup(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]],
                 ref_img_bgr: Optional[np.ndarray] = None) -> np.ndarray:
    if reg.get('beautygan') is not None:
        return reg['beautygan'](img_bgr, ref=ref_img_bgr)
    # Fallback: gentle contrast + saturation boost
    out = adjust_brightness_contrast(img_bgr, brightness=0.02, contrast=0.15)
    out = color_jitter_hsv(out, hue_shift=0, sat_scale=1.08, val_scale=1.02)
    return out


def apply_accessory(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]],
                    kind: str = 'glasses', overlay_rgba: Optional[np.ndarray] = None,
                    bbox: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    if reg.get('infogan') is not None:
        return reg['infogan'](img_bgr, attr=kind)
    # Fallback: overlay
    if overlay_rgba is not None and bbox is not None:
        return overlay_png_rgba(img_bgr, overlay_rgba, bbox)
    return img_bgr

def apply_prof_change(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]],
                      yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> np.ndarray:
    """
    Use EG3D to render a new face pose (side profile, tilt, etc).
    
    Args:
        img_bgr: Input face image (BGR).
        reg: Dictionary of GAN callables (expects 'eg3d' if available).
        yaw, pitch, roll: Rotation angles in radians or degrees (depends on EG3D wrapper).
    
    Returns:
        np.ndarray: Face image in the new pose.
    """
    if reg.get('eg3d') is not None:
        return reg['eg3d'](img_bgr, yaw=yaw, pitch=pitch, roll=roll)

    # Fallback: naive 2D warp (not true 3D, just an approximation)
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), np.degrees(yaw), 1.0)
    out = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)
    return out

def apply_pose_change(img_bgr: np.ndarray,
                       reg: Dict[str, Optional[GanCallable]],
                       include_profile=True,
                       realism_after_each=True,
                       driver_for_pose: Optional[np.ndarray] = None,
                       yaw_range=(-0.35, 0.35)) -> Dict[str, list]:
    
    out = {'pose': [], 'expr': [], 'all': []}
    
    if reg.get('discogan') is not None:
        # GAN-driven pose/profile (DiscoFaceGAN)
        discogan = reg.get('discogan')
        if discogan is not None:
            yaw = float(np.random.uniform(*yaw_range))
            pgan = discogan(img_bgr, yaw=yaw, pitch=0, roll=0)
            pgan = apply_realism(pgan, reg) if realism_after_each else pgan
            out['pose'].append(pgan)
            out['all'].append(pgan)

    return out
"""
def apply_pose_change(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]],
                       driver_img_bgr: Optional[np.ndarray] = None,
                       yaw: Optional[float] = None) -> np.ndarray:
    if reg.get('pose_gan') is not None:
        return reg['pose_gan'](img_bgr, driver=driver_img_bgr, yaw=yaw)
    # Fallback: classical yaw skew
    direction = 'left' if (yaw is not None and yaw < 0) else 'right'
    strength = min(0.35, abs(yaw) if yaw is not None else 0.22)
    return yaw_warp(img_bgr, direction=direction, strength=strength)
"""

def apply_expression(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]],
                     expr: str = 'smile', intensity: float = 0.7) -> np.ndarray:
    if reg.get('expregan') is not None:
        return reg['expregan'](img_bgr, expr=expr, intensity=float(np.clip(intensity, 0, 1)))
    # Fallback: slight mouth/lip contrast tweak to hint expression
    out = img_bgr.copy()
    h, w = out.shape[:2]
    mouth_box = (int(0.25*w), int(0.6*h), int(0.75*w), int(0.88*h))
    x1, y1, x2, y2 = mouth_box
    roi = out[y1:y2, x1:x2]
    roi = adjust_brightness_contrast(roi, brightness=0.03, contrast=0.22)
    out[y1:y2, x1:x2] = roi
    return out


def apply_dagan(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]]) -> np.ndarray:
    if reg.get('dagan') is not None:
        return reg['dagan'](img_bgr)
    return img_bgr


def apply_realism(img_bgr: np.ndarray, reg: Dict[str, Optional[GanCallable]]) -> np.ndarray:
    if reg.get('realism_gan') is not None:
        return reg['realism_gan'](img_bgr)
    return realism_enhance_lite(img_bgr)

# --------------------------------
# Orchestrator
# --------------------------------

def make_augmented_faces_full(
    face_bgr: np.ndarray,
    *,
    gan_registry: Optional[Dict[str, Optional[GanCallable]]] = None,
    n_geo_photo: int = 6,
    include_pose: bool = True,
    include_expr: bool = True,
    include_hair: bool = True,
    include_makeup: bool = True,
    include_accessory: bool = True,
    include_profiles: bool = True,
    accessory_rgba: Optional[np.ndarray] = None,
    accessory_bbox: Optional[Tuple[int, int, int, int]] = None,
    ref_for_hair: Optional[np.ndarray] = None,
    ref_for_makeup: Optional[np.ndarray] = None,
    driver_for_pose: Optional[np.ndarray] = None,
    expr_label: str = 'smile',
    expr_intensity: float = 0.7,
    realism_after_each: bool = True,
    master_seed: Optional[int] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Build a rich augmentation pack and return a dict of lists by category.

    All outputs are uint8 BGR. By default, each generated image is passed
    through a realism enhancer (GAN if provided, else a lite enhancer).
    """
    rng = np.random.default_rng(master_seed)
    reg = gan_registry or {}

    out: Dict[str, List[np.ndarray]] = {
        'geo_photo': [],
        'pose': [],
        'expr': [],
        'hair': [],
        'makeup': [],
        'accessory': [],
        'dagan': [],
        'profiles': [],
        'all': []
    }

    base = face_bgr.copy()

    # 1) Geometric + Photometric
    for i in range(n_geo_photo):
        img = geometric_photometric_bundle(base, seed=int(rng.integers(0, 1<<31)))
        if realism_after_each:
            img = apply_realism(img, reg)
        out['geo_photo'].append(img)
        out['all'].append(img)

    # 2) Pose/profile
    if include_pose:
        # classical pair
        for p in pose_profile_variants(base):
            p2 = apply_realism(p, reg) if realism_after_each else p
            out['pose'].append(p2)
            out['all'].append(p2)
        # GAN-driven (optional)
        if reg.get('discogan') is not None:
            pgan = apply_pose_change(base, reg, driver_img_bgr=driver_for_pose, yaw=float(rng.uniform(-0.35, 0.35)))
            pgan = apply_realism(pgan, reg) if realism_after_each else pgan
            out['pose'].append(pgan)
            out['all'].append(pgan)

    if include_profiles:
        # classical pair
        for p in pose_profile_variants(base):
            p2 = apply_realism(p, reg) if realism_after_each else p
            out['pose'].append(p2)
            out['all'].append(p2)

        # GAN-driven (optional)
        if reg.get('eg3d') is not None:
            pgan = apply_pose_change(
                base,
                reg,
                driver_img_bgr=driver_for_pose,
                yaw=float(rng.uniform(-0.35, 0.35))
            )
            pgan = apply_realism(pgan, reg) if realism_after_each else pgan
            out['pose'].append(pgan)
            out['all'].append(pgan)

    # 3) Expression
    if include_expr:
        e = apply_expression(base, reg, expr=expr_label, intensity=expr_intensity)
        e = apply_realism(e, reg) if realism_after_each else e
        out['expr'].append(e)
        out['all'].append(e)

    # 4) Hair
    if include_hair:
        hchg = apply_hair_change(base, reg, ref_img_bgr=ref_for_hair)
        hchg = apply_realism(hchg, reg) if realism_after_each else hchg
        out['hair'].append(hchg)
        out['all'].append(hchg)

    # 5) Makeup
    if include_makeup:
        mk = apply_makeup(base, reg, ref_img_bgr=ref_for_makeup)
        mk = apply_realism(mk, reg) if realism_after_each else mk
        out['makeup'].append(mk)
        out['all'].append(mk)

    # 6) Accessories
    if include_accessory:
        acc = apply_accessory(base, reg, kind='glasses', overlay_rgba=accessory_rgba, bbox=accessory_bbox)
        acc = apply_realism(acc, reg) if realism_after_each else acc
        out['accessory'].append(acc)
        out['all'].append(acc)

    # 7) DAGAN generic augmentation (optional)
    if reg.get('dagan') is not None:
        d = apply_dagan(base, reg)
        d = apply_realism(d, reg) if realism_after_each else d
        out['dagan'].append(d)
        out['all'].append(d)

    return out

# --------------------------------
# Minimal legacy API compatibility
# --------------------------------

def make_five_augmented_faces(face_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Backward-compatible helper returning 5 variants:
      [original, gamma(0.7), horizontal flip, yaw-left, yaw-right]
    """
    aug = []
    aug.append(face_bgr.copy())
    aug.append(gamma_adjust_uint8(face_bgr, gamma=0.7))
    aug.append(horizontal_flip(face_bgr))
    aug.append(yaw_warp(face_bgr, direction='left', strength=0.22))
    aug.append(yaw_warp(face_bgr, direction='right', strength=0.22))
    return aug

# --------------------------------
# Convenience: end-to-end one-shot for a single face crop
# --------------------------------

def generate_face_aug_pack(
    face_bgr: np.ndarray,
    gan_config: Optional[Dict[str, Any]] = None,
    accessory_png_path: Optional[str] = None,
    accessory_bbox: Optional[Tuple[int, int, int, int]] = None,
    ref_hair_img: Optional[np.ndarray] = None,
    ref_makeup_img: Optional[np.ndarray] = None,
    driver_pose_img: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
) -> Dict[str, List[np.ndarray]]:
    """
    Build the GAN registry (from callables or config) and run make_augmented_faces_full.
    """
    reg = build_gan_registry(gan_config)
    accessory_rgba = None
    if accessory_png_path and os.path.exists(accessory_png_path):
        overlay = cv2.imread(accessory_png_path, cv2.IMREAD_UNCHANGED)
        if overlay is not None and overlay.shape[-1] == 4:
            accessory_rgba = overlay
    pack = make_augmented_faces_full(
        face_bgr,
        gan_registry=reg,
        n_geo_photo=6,
        include_pose=True,
        include_expr=True,
        include_hair=True,
        include_makeup=True,
        include_accessory=True,
        accessory_rgba=accessory_rgba,
        accessory_bbox=accessory_bbox,
        ref_for_hair=ref_hair_img,
        ref_for_makeup=ref_makeup_img,
        driver_for_pose=driver_pose_img,
        expr_label='smile',
        expr_intensity=0.7,
        realism_after_each=True,
        master_seed=seed,
    )
    return pack
