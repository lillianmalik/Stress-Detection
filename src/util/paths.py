import os

# project root = parent of /src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "WESAD")

PROCESSED_FEATURES_PATH = os.path.join(PROJECT_ROOT, "processed_features.csv")
WINDOWED_FEATURES_PATH = os.path.join(PROJECT_ROOT, "windowed_features.csv")