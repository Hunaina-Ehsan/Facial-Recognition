import cv2 
import time 
import numpy as np 
import torch 
from collections import deque 
from PIL import Image 
from facenet_pytorch import InceptionResnetV1 
from ultralytics import YOLO 
from torchvision import transforms 
from collections import defaultdict, deque 
import statistics
import concurrent.futures
from lsh_helper import build_lsh_index, lsh_lookup
from utils import is_dark, adjust_gamma, log_lsh_query, get_dynamic_threshold_cosface
from embeddings import load_context_maps_from_db
import yaml
import os
import logging
from build_exe import resource_path

config_path = resource_path("config/config.yaml")


with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Create logs directory if not exists
os.makedirs("logs", exist_ok=True)

# Configure logging
logging.basicConfig(
    filename="logs/app.log",
    filemode="a",
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO
)

# Optional: also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

output_path = 'outputs/yolo_avengers_out_wah.mp4'

# ==== CONFIG ==== #
FRAME_SKIP = cfg["frame_skip"]
MAX_LAG_SEC = cfg["max_lag_sec"] # if behind > this, fast forward #

known_db_path = cfg["paths"]["known_db"]
video_path = cfg["paths"]["video"]

#  ==== Device ==== 
device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu') 
logging.info(f"Using device: {device}")

transform = transforms.Compose([ transforms.Resize((160, 160)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ]) 

yolo_model_path = resource_path(cfg["paths"]["yolov8_model"])
# ==== Models ==== 
yolo = YOLO(yolo_model_path).to(device) 
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device) 

# Global metrics collectors
timings = {
    "embedding": [],
    "lsh_lookup": [],
    "detection": [],
    "frame_total": []
}

# Database parameters
HOST = cfg["database"]["host"]
USERNAME = cfg["database"]["user"]
PASSWORD = cfg["database"]["password"]
DB_NAME = cfg["database"]["db_name"]

NUM_PLANES = cfg["num_planes"]
NUM_TABLES = cfg["num_tables"]

confidence_thresh = cfg["confidence_thresh"]  
l2_thresh = cfg["l2_thresh"]
gamma = cfg["gamma_dark"]

# Load them
known_context_maps = load_context_maps_from_db(host=HOST, user=USERNAME, password=PASSWORD, database=DB_NAME)
context_vectors = {label: maps["attn"] for label, maps in known_context_maps.items()}

context_vectors = {label: maps["attn"] for label, maps in known_context_maps.items()}

planes_list, buckets_list = build_lsh_index(
    context_vectors,
    num_planes=NUM_PLANES,   # try 12–16
    n_tables=NUM_TABLES,      # try 4–8 for better recall
    seed=42
)

# =============================
# Real-Time Face Recognition
# =============================
import concurrent.futures

def run_realtime_face_recognition(known_db_path): 
    cap = cv2.VideoCapture(0) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) 
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720) 
    
    font = cv2.FONT_HERSHEY_SIMPLEX 
    frame_count = 0 
    label_history = defaultdict(lambda: deque(maxlen=5)) 
    last_faces = [] 
    last_embedding = None  # track last embedding for duplicate skipping

    # Motion detection init
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Lag-handling variables 
    fps = 30 
    frame_interval = 1.0 / fps 
    last_time = time.time() 

    while True: 
        ret, frame = cap.read() 
        if not ret: 
            break 
        
        # Motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #diff = cv2.absdiff(prev_gray, gray)
        #non_zero_count = np.count_nonzero(diff)
        #if non_zero_count < 5000:  # skip if no motion
        #    prev_gray = gray
        #    cv2.imshow("Real-Time YOLOv8 Face Recognition", frame)
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break
        #    continue

        # --- LAG HANDLING --- 
        now = time.time() 
        elapsed = now - last_time 
        if elapsed < frame_interval: 
            time.sleep(frame_interval - elapsed) 
            now = time.time() 
        elif elapsed > 0.5: 
            skip_count = int(elapsed / frame_interval) - 1 
            for _ in range(skip_count): 
                cap.read() 
                now = time.time() 
            last_time = now 
            
        if is_dark(frame): 
            frame = adjust_gamma(frame, gamma=gamma) 
            
        current_faces = [] 
        if frame_count % FRAME_SKIP == 0: 
            results = yolo.predict(source=frame, conf=0.5, verbose=False)[0] 
            if results.boxes is not None: 
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_faces = []
                    for box in results.boxes: 
                        conf = float(box.conf) 
                        if conf < confidence_thresh: 
                            continue 
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                        face_width, face_height = x2 - x1, y2 - y1 
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue

                        # Threaded face processing
                        future_faces.append(executor.submit(
                            process_face, face_crop, x1, y1, x2, y2,
                            label_history, lsh_lookup, l2_thresh,
                            face_width, face_height
                        ))

                    for future in concurrent.futures.as_completed(future_faces):
                        try:
                            result = future.result()
                            if result:
                                (x1, y1, x2, y2), smoothed_label, best_score, emb = result
                                current_faces.append(((x1, y1, x2, y2), smoothed_label, best_score))
                                last_embedding = emb
                        except Exception:
                            logging.error("Threaded face error:", exc_info=True)

            last_faces = current_faces.copy() 
        
        # Draw detections
        for (x1, y1, x2, y2), label, score in last_faces: 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2) 
            cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10),
                        font, 0.6, (255, 255, 255), 2) 
        
        cv2.imshow("Real-Time YOLOv8 Face Recognition", frame) 
        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break 
        
        frame_count += 1 
        prev_gray = gray
    
    cap.release()
    cv2.destroyAllWindows()


def process_face(face_crop, x1, y1, x2, y2, label_history, lsh_lookup, l2_thresh, face_width, face_height):
    try:
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
        face_tensor = transform(face_pil).unsqueeze(0).to(device)

        # Get embedding for face
        with torch.no_grad():
            emb = facenet(face_tensor).cpu().numpy().astype(np.float32).flatten()

        # Skip if similar to the last embedding
        if last_embedding is not None:
            sim = np.dot(emb, last_embedding) / (np.linalg.norm(emb) * np.linalg.norm(last_embedding))
            if sim > 0.95:
                return None

        # Lookup in LSH
        best_label, best_score = lsh_lookup(
            emb, planes_list, buckets_list, context_vectors,
            l2_thresh=l2_thresh, probe_all_tables=True
        )

        # Dynamic threshold based on face size
        dyn_thresh = get_dynamic_threshold_cosface(face_width, face_height)
        final_label = best_label if best_score > dyn_thresh else "Unknown"

        # Track label history to avoid flickering
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        label_history[center].append(final_label)
        smoothed_label = max(set(label_history[center]), key=label_history[center].count)

        return ((x1, y1, x2, y2), smoothed_label, best_score, emb)

    except Exception as e:
        logging.error("Error processing face:", exc_info=True)

        return None

last_embedding = None

# ========================
# Video Processing with Threading
# ========================

def run_video_face_recognition(video_path, known_db_path, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Could not open video: {video_path}")
        return

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_count = 0
    label_history = defaultdict(lambda: deque(maxlen=5))
    last_faces = []

    # Motion detection init
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        start_frame = time.time()
        current_faces = []
        if frame_count % FRAME_SKIP == 0:
            if is_dark(frame):
                frame = adjust_gamma(frame, gamma=1.8)

            t0 = time.time()
            results = yolo.predict(source=frame, conf=confidence_thresh, verbose=False)[0]
            timings["detection"].append(time.time() - t0)

            if results.boxes is not None:
                # Using threading to process multiple faces in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_faces = []
                    for box in results.boxes:
                        conf = float(box.conf)
                        if conf < confidence_thresh:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        face_width, face_height = x2 - x1, y2 - y1
                        face_crop = frame[y1:y2, x1:x2]
                        if face_crop.size == 0:
                            continue

                        # Submit each face processing task to the thread pool
                        future_faces.append(executor.submit(
                            process_face, face_crop, x1, y1, x2, y2, label_history,
                            lsh_lookup, l2_thresh, face_width, face_height
                        ))

                    # Collect results from threads
                    for future in concurrent.futures.as_completed(future_faces):
                        result = future.result()
                        if result:
                            (x1, y1, x2, y2), smoothed_label, best_score, emb = result
                            current_faces.append(((x1, y1, x2, y2), smoothed_label, best_score))
                            last_embedding = emb

            last_faces = current_faces.copy()

        for (x1, y1, x2, y2), label, score in last_faces:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({score:.2f})", (x1, y1 - 10), font, 0.6, (255, 255, 255), 2)

        if writer is not None:
            writer.write(frame)
        cv2.imshow("Video Face Recognition", frame)

        timings["frame_total"].append(time.time() - start_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1
        prev_gray = gray

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    logging.info(f"----------PROCESSING TIME----------")
    for k, vals in timings.items():
        if vals:
            logging.info(f"{k:12s} mean={statistics.mean(vals)*1000:.2f} ms   std={statistics.pstdev(vals)*1000:.2f} ms   n={len(vals)}")


# === Example usage ===
# run_video_face_recognition(video_path, known_db_path, output_path=output_path)

#run_realtime_face_recognition('hunaina_db')

# Added 2 GANs in code, acc went up teeny tiny bit