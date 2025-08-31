import cv2
from utils import is_dark, adjust_gamma, get_embedding, cosine_similarity, l2_distance
from PIL import Image
import torch
import yaml
from torchvision import transforms 
from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1 
from embeddings import load_context_maps_from_db
from lsh_helper import lsh_lookup, build_lsh_index
from build_exe import resource_path
import logging
import os


config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

# Database parameters
HOST = cfg["database"]["host"]
USERNAME = cfg["database"]["user"]
PASSWORD = cfg["database"]["password"]
DB_NAME = cfg["database"]["db_name"]

image = resource_path(cfg["paths"]["img_path"])

device = torch.device(cfg["device"] if torch.cuda.is_available() else 'cpu') 
yolo_model_path = resource_path(cfg["paths"]["yolov8_model"])

NUM_PLANES = cfg["num_planes"]
NUM_TABLES = cfg["num_tables"]

out_path = cfg["paths"]["out_img"]


# ==== Models ==== 
yolo = YOLO(yolo_model_path).to(device) 
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device) 
transform = transforms.Compose([ transforms.Resize((160, 160)), transforms.ToTensor(), transforms.Normalize([0.5], [0.5]) ]) 

context_maps = load_context_maps_from_db(HOST, USERNAME, PASSWORD, DB_NAME)
context_vectors = {label: maps["attn"] for label, maps in context_maps.items()}

planes_list, buckets_list = build_lsh_index(
    context_vectors,
    num_planes=NUM_PLANES,   # try 12–16
    n_tables=NUM_TABLES,      # try 4–8 for better recall
    seed=42
)

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

def recognize_faces_in_image(image_path, known_embeddings, conf_thresh=0.5, l2_thresh=1.0, output_path=None):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Image could not be read.")
        return [], None

    # Brighten if dark
    if is_dark(frame):
        frame = adjust_gamma(frame, gamma=1.5)

    # Resize if too large
    MAX_RES = 720
    h, w = frame.shape[:2]
    if max(h, w) > MAX_RES:
        scale = MAX_RES / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Convert to RGB for YOLO
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = yolo(frame_rgb)[0]

    predictions = []

    if results.boxes is not None and len(results.boxes) > 0:
        for box in results.boxes:
            conf = float(box.conf.item())
            if conf < conf_thresh:
                continue

            # Extract coordinates for this specific detection
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])

            # Crop face
            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            # Convert to tensor & get embedding
            face_pil = Image.fromarray(face_crop)
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            emb = get_embedding(face_tensor)

            # Compare with known embeddings
            best_score = -1
            best_label = "Unknown"
            best_label, best_score = lsh_lookup(emb, planes_list, buckets_list, context_vectors, l2_thresh=l2_thresh, probe_all_tables=True)


            # Store prediction
            predictions.append({
                "label": best_label,
                "score": best_score,
                "box": (x1, y1, x2, y2)
            })

            # Draw box & label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{best_label} ({best_score:.2f})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2)

    if not predictions:
        logging.info("No face detected in input image.")
    else:
        for i, pred in enumerate(predictions):
            logging.info(f"[{i+1}] {pred['label']} ({pred['score']:.2f})")

    # Save if requested
    if output_path:
        cv2.imwrite(output_path, frame)
        logging.info(f"[✔] Result saved as {output_path}")

    return predictions, frame

recognize_faces_in_image(image, context_maps, conf_thresh=0.5, l2_thresh= 1, output_path=out_path)
