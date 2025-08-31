from yolop3 import run_realtime_face_recognition
import yaml
from build_exe import resource_path

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

known_db = cfg["paths"]["known_db"]

if __name__ == "__main__":
    run_realtime_face_recognition(known_db_path=known_db)
