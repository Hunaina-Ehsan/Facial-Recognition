from yolop3 import run_video_face_recognition
import yaml
from build_exe import resource_path

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

known_db = cfg["paths"]["known_db"]
inp_vid = resource_path(cfg["paths"]["video"])
out_vid = cfg["paths"]["out_vid"]

if __name__ == "__main__":
    run_video_face_recognition(inp_vid, known_db_path=known_db, output_path=out_vid)
