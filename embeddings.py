import os
import pickle
import cv2
import numpy as np
from sklearn.decomposition import PCA
import mysql.connector
import json
from ultralytics import YOLO
import torch
from sklearn.cluster import KMeans
import yaml
from aug_helpers import find_one_image_named_1, generate_face_aug_pack
from utils import face_to_embedding
from build_exe import resource_path

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)


yolo_model_path = resource_path(cfg["paths"]["yolov8_model"])
device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu') 

N_PCA = cfg["n_pca"]
N_PROTO = cfg["n_proto"]

yolo = YOLO(yolo_model_path).to(device) 

# ---------- UPDATED: build_known_embeddings ----------


def build_known_embeddings(known_db_path):
    """
    Build per-person embeddings by:
      - Reading only the image named '1.*' in each person's folder
      - Detecting the face and cropping once
      - Generating 5 augmented variants (original, gamma, flip, yaw-left, yaw-right)
      - Averaging embeddings of those 5 to get a single vector per person
    """

    print("[INFO] Building embeddings (single '1' image per person with 5 augmentations)...")
    embeddings = {}

    for person_name in os.listdir(known_db_path):
        folder = os.path.join(known_db_path, person_name)
        if not os.path.isdir(folder):
            continue

        one_img_path = find_one_image_named_1(folder)
        if not one_img_path:
            print(f"[WARN] No '1.*' image found for {person_name}, skipping.")
            continue

        frame = cv2.imread(one_img_path)
        if frame is None:
            print(f"[WARN] Could not read {one_img_path} for {person_name}, skipping.")
            continue

        # Detect face once
        results = yolo(frame)[0]
        if results.boxes is None or len(results.boxes) == 0:
            print(f"[WARN] No face found in {one_img_path} for {person_name}, skipping.")
            continue

        # Take the highest-confidence box
        boxes = results.boxes
        confs = boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())

        # Crop safely
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w-1, x2), min(h-1, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print(f"[WARN] Empty face crop in {one_img_path} for {person_name}, skipping.")
            continue

        # Make exactly 5 variants
        pack = generate_face_aug_pack(face_crop)   # returns dict of lists
        variants = pack['all']                     # unified list of augmented variants


        # Embeddings -> mean
        embs = [face_to_embedding(v) for v in variants]
        mean_emb = np.mean(np.stack(embs), axis=0)

        embeddings[person_name] = mean_emb

    return embeddings


# ---------- UPDATED: build_known_context_maps ----------

def build_known_context_maps(known_db_path, n_pca=N_PCA, n_proto=N_PROTO):
    """
    Build rich context maps:
      - mean embedding
      - attention-weighted embedding
      - top PCA directions
      - covariance matrix
      - representative prototypes
    """

    print("[INFO] Building context maps (with PCA, covariance, prototypes)...")
    context_maps = {}

    for person_name in os.listdir(known_db_path):
        folder = os.path.join(known_db_path, person_name)
        if not os.path.isdir(folder):
            continue

        one_img_path = find_one_image_named_1(folder)
        if not one_img_path:
            print(f"[WARN] No '1.*' image for {person_name}, skipping.")
            continue

        frame = cv2.imread(one_img_path)
        if frame is None:
            print(f"[WARN] Could not read {one_img_path}, skipping {person_name}.")
            continue

        results = yolo(frame)[0]
        if results.boxes is None or len(results.boxes) == 0:
            print(f"[WARN] No face in {one_img_path}, skipping {person_name}.")
            continue

        boxes = results.boxes
        confs = boxes.conf.detach().cpu().numpy()
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w-1, x2), min(h-1, y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            print(f"[WARN] Empty crop for {person_name}, skipping.")
            continue

        # Get augmented variants
        pack = generate_face_aug_pack(face_crop)   # returns dict of lists
        variants = pack['all']                     # unified list of augmented variants

        # Embeddings
        all_embs = np.stack([face_to_embedding(v) for v in variants])  # shape (N, D)

        # Mean
        mean_emb = np.mean(all_embs, axis=0)

        # Attention-weighted
        sims = np.dot(all_embs, mean_emb) / (
            np.linalg.norm(all_embs, axis=1) * (np.linalg.norm(mean_emb) + 1e-8) + 1e-8
        )
        weights = sims / (sims.sum() + 1e-8)
        attn_emb = np.average(all_embs, axis=0, weights=weights)

        # PCA
        pca_vecs = None
        if all_embs.shape[0] >= n_pca:
            pca = PCA(n_components=n_pca)
            pca.fit(all_embs)
            pca_vecs = pca.components_.astype(np.float32)
        else:
            pca_vecs = np.zeros((n_pca, all_embs.shape[1]), dtype=np.float32)

        # Covariance matrix
        cov = np.cov(all_embs.T).astype(np.float32)

        # Prototypes (cluster centers)
        if all_embs.shape[0] >= n_proto:
            kmeans = KMeans(n_clusters=n_proto, n_init=5, random_state=42)
            kmeans.fit(all_embs)
            prototypes = kmeans.cluster_centers_.astype(np.float32)
        else:
            prototypes = all_embs.astype(np.float32)

        context_maps[person_name] = {
            "mean": mean_emb.astype(np.float32),
            "attn": attn_emb.astype(np.float32),
            "pca_vecs": pca_vecs,
            "covariance": cov,
            "prototypes": prototypes
        }

    return context_maps



# ==============================
# Save context maps to MySQL
# ==============================
def save_context_maps_to_db(context_maps, host, user, password, database):
    """
    Saves context maps {person: {"mean": np.array, "attn": np.array}} to MySQL DB.
    """
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS context_maps (
            person VARCHAR(255) PRIMARY KEY,
            mean_vec JSON,
            attn_vec JSON
        )
    """)

    for person, maps in context_maps.items():
        mean_vec = json.dumps(maps["mean"].tolist())
        attn_vec = json.dumps(maps["attn"].tolist())
        cursor.execute("""
            REPLACE INTO context_maps (person, mean_vec, attn_vec)
            VALUES (%s, %s, %s)
        """, (person, mean_vec, attn_vec))

    conn.commit()
    cursor.close()
    conn.close()
    print(f"[INFO] Saved {len(context_maps)} context maps to MySQL")


# ==============================
# Load context maps from MySQL
# ==============================
def load_context_maps_from_db(host, user, password, database):
    """
    Loads context maps into {person: {"mean": np.array, "attn": np.array}} format.
    """
    conn = mysql.connector.connect(
        host=host, user=user, password=password, database=database
    )
    cursor = conn.cursor()
    cursor.execute("SELECT person, mean_vec, attn_vec FROM context_maps")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    context_maps = {}
    for person, mean_json, attn_json in rows:
        context_maps[person] = {
            "mean": np.array(json.loads(mean_json), dtype=np.float32),
            "attn": np.array(json.loads(attn_json), dtype=np.float32)
        }

    print(f"[INFO] Loaded {len(context_maps)} context maps from MySQL")
    return context_maps