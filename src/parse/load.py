import os       # for file parsing
import pickle   # for .pkl files in raw dataset
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)  # for outdated NumPy version


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