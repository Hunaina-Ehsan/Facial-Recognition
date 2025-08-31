
from embeddings import build_known_context_maps, save_context_maps_to_db
import yaml
from build_exe import resource_path

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

known_db_path = cfg["paths"]["known_db"]

HOST = cfg["database"]["host"]
USERNAME = cfg["database"]["user"]
PASSWORD = cfg["database"]["password"]
DB_NAME = cfg["database"]["db_name"]

# Build context maps
known_context_maps = build_known_context_maps(known_db_path)

# Save them
save_context_maps_to_db(known_context_maps, host=HOST, user=USERNAME, password=PASSWORD, database=DB_NAME)
