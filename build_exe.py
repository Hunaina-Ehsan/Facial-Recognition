import os, sys, yaml

def resource_path(relative_path):
    """Get absolute path to resource, works in dev and PyInstaller exe"""
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

config_path = resource_path("config/config.yaml")

with open(config_path, "r") as f:
    config = yaml.safe_load(f)
