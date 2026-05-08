# load.py
import os           # for file parsing
import pickle       # for .pkl files in raw dataset
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # for outdated NumPy version

from src.util.paths import DATA_DIR, RAW_DATA_PATH
import subprocess   # for automatically downloading the WESAD dataset
import zipfile

KAGGLE_DATASET = "orvile/wesad-wearable-stress-affect-detection-dataset"


def download_wesad():
    # Downloads and extracts WESAD dataset (if not already present)

    # already exists -> skip
    if os.path.exists(RAW_DATA_PATH):
        print("[DATA] Dataset already exists. Skipping download.")
        return RAW_DATA_PATH

    os.makedirs(DATA_DIR, exist_ok=True)

    print("[DATA] Downloading dataset from Kaggle...")

    subprocess.run([ "kaggle", "datasets", "download", "-d", KAGGLE_DATASET, "-p", DATA_DIR ], check=True)

    zip_path = os.path.join(DATA_DIR, "wesad-wearable-stress-affect-detection-dataset.zip")

    if not os.path.exists(zip_path):
        raise RuntimeError("Expected Kaggle ZIP not found. Check dataset name.")

    print("[DATA] Extracting dataset...")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    os.remove(zip_path)

    # Kaggle datasets often nest folders, so we rely on your loader robustness
    if not os.path.exists(RAW_DATA_PATH):
        raise RuntimeError(
            f"Dataset extracted but expected path not found: {RAW_DATA_PATH}"
        )

    print("[DATA] Dataset ready.")
    return RAW_DATA_PATH

def load_subject(file_path):
    # loads a single WESAD subject .pkl file
    # args:
    #   file_path (str): path to SX.pkl file
    # returns:
    #   dict: loaded subject data
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    # basic validation
    if "signal" not in data or "label" not in data:
        raise ValueError(f"Invalid WESAD format in {file_path}")

    return data


def load_all_subjects(data_dir):
    # load all valid WESAD subjects from a directory
    # skips:
    #   - non .pkl files
    #   - corrupted files
    # args:
    #   data_dir (str): path to folder containing SX.pkl files
    # returns:
    #   list: list of subject dictionaries

    subjects = []

    for folder in sorted(os.listdir(data_dir)):
        subject_path = os.path.join(data_dir, folder)

        if not os.path.isdir(subject_path):
            continue

        pkl_file = os.path.join(subject_path, f"{folder}.pkl")

        if not os.path.exists(pkl_file):
            print(f"[WARNING] Missing {folder}.pkl")
            continue

        try:
            subject = load_subject(pkl_file)
            subject["subject_id"] = folder
            subjects.append(subject)
            print(f"Loaded {folder}")

        except Exception as e:
            print(f"[WARNING] Skipping {folder}: {e}")

    if len(subjects) == 0:
        raise RuntimeError("No subjects loaded.")

    return subjects