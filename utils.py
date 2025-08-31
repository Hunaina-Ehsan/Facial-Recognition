import torch
from facenet_pytorch import InceptionResnetV1 
import cv2
from PIL import Image
from torchvision import transforms 
import numpy as np
import json
import yaml
import os
import sys
from build_exe import resource_path

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu') 

facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device) 

transform = transforms.Compose([ transforms.Resize((160, 160)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ]) 

DYNAMIC_LARGE = cfg["d_thresh_l"]
DYNAMIC_MEDIUM = cfg["d_thresh_m"]
DYNAMIC_SMALL = cfg["d_thresh_s"]
DYNAMIC_XSMALL = cfg["d_thresh_xs"]

MARGIN = cfg["margin"]
SCALE = cfg["scale"]

BRIGHTNESS_THRESH = cfg["brightness_thresh"]

def get_embedding(face_tensor): 
    with torch.no_grad(): 
        return facenet(face_tensor).cpu().numpy().flatten() 
    
    
def get_dynamic_threshold(face_width, face_height): 
    size = min(face_width, face_height) 
    if size > 150: 
        return DYNAMIC_LARGE
    elif size > 100: 
        return DYNAMIC_MEDIUM
    elif size > 60: 
        return DYNAMIC_SMALL
    else: return DYNAMIC_XSMALL

def get_dynamic_threshold_cosface(face_width, face_height, margin=MARGIN):
    # Calculate the size of the face as the minimum of width and height
    size = min(face_width, face_height)
    
    # Apply margin adjustments depending on face size
    if size > 150:
        # Larger faces might be more confident, reduce the threshold
        threshold = DYNAMIC_LARGE - margin * 0.1
    elif size > 100:
        threshold = DYNAMIC_MEDIUM - margin * 0.05
    elif size > 60:
        threshold = DYNAMIC_SMALL - margin * 0.05
    else:
        # For smaller faces, keep a lower threshold due to more noise
        threshold = DYNAMIC_XSMALL - margin * 0.05
        
    # Make sure threshold stays within a reasonable range
    threshold = max(0.2, min(threshold, 0.5))
    
    return threshold
 
# === Cosine Similarity ===
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def l2_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)
    
def face_to_embedding(face_bgr):
    """
    Convert BGR face image to embedding (uses your existing global transform/facenet).
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_pil = Image.fromarray(face_rgb)
    face_tensor = transform(face_pil).unsqueeze(0).to(device)
    
    # Detach the tensor before converting to numpy
    with torch.no_grad(): 
        embedding = facenet(face_tensor)
    return embedding.detach().cpu().numpy().flatten()  # Now detaching before converting to numpy



# ========================== Logging ===========================
def log_lsh_query(embedding, planes_list, buckets_list):
    """
    Debug helper: show which hash buckets a given embedding falls into.
    """
    query_planes = []
    query_buckets = []

    for table_idx, planes in enumerate(planes_list):
        # hash = sign(embedding · plane)
        signs = (embedding @ planes.T) >= 0
        hash_str = ''.join(['1' if s else '0' for s in signs])
        query_planes.append(signs.astype(int).tolist())
        query_buckets.append(hash_str)

    log_data = {
        "planes": query_planes,       # 0/1 for each plane
        "bucket_ids": query_buckets   # string like "101001"
    }

    # Print once in a while (not every frame)
    #print(json.dumps(log_data, indent=2))

    # (optional) save to file for visualization later
    #with open("lsh_debug2.jsonl", "a") as f:
    #    f.write(json.dumps(log_data) + "\n")

 

# ========= Lighting Adjustments ========= 

def adjust_gamma(image, gamma=1.0): 
    invGamma = 1.0 / gamma 
    table = np.array([((i / 255.0) ** invGamma) * 255 
                      for i in np.arange(0, 256)]).astype("uint8") 
    return cv2.LUT(image, table) 

def is_dark(frame, brightness_thresh=BRIGHTNESS_THRESH): 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    mean_brightness = np.mean(gray) 
    return mean_brightness < brightness_thresh 

# --- utils ---
def _normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def _cosine(a, b):
    return float(np.dot(a, b))

# --- cosface style scoring ---
def _cosface_score(a, b, margin=MARGIN, scale=SCALE):
    """
    CosFace-style similarity:
    s * (cos(theta) - m)
    where cos(theta) = normalized dot product
    """
    a = _normalize(a)
    b = _normalize(b)
    cos_theta = float(np.dot(a, b))
    return scale * (cos_theta - margin)


def _l2_sq(a, b):
    return float(np.sum((a - b) ** 2))  # squared L2, no sqrt

def _generate_random_planes(num_planes, dim, rng):
    return rng.randn(num_planes, dim).astype(np.float32)

def _hash_vector(vec, planes):
    projections = planes @ vec
    bits = (projections > 0).astype(np.uint8)
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

config_path = resource_path("config/config.yaml")