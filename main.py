# main.py
import pandas as pd

from src.parse.load import load_all_subjects, download_wesad
from src.util.paths import RAW_DATA_PATH, PROCESSED_FEATURES_PATH
from src.features.featureExtraction import extract_features
from src.modeling.models import model_data


# ==================================================================== CONFIG

DOWNSAMPLE_FACTOR = 70                  # adjust if needed (for speed)

# ==================================================================== HELPER FUNCTIONS
def subject_to_dataframe(subject, downsample_factor=DOWNSAMPLE_FACTOR):
    # converts one subject's data into a clean dataframe

    chest = subject['signal']['chest']
    labels = subject['label']

    df = pd.DataFrame({
        'ECG': chest['ECG'].flatten(),  # flatten bc of weird columns
        'EDA': chest['EDA'].flatten(),
        'TEMP': chest['Temp'].flatten(),
        'label': labels
    })

    df['subject_id'] = subject['subject_id']
    # keep only baseline (1) vs stress (2)
    df = df[df['label'].isin([1, 2])]

    # downsample (for speed)
    df = df.iloc[::downsample_factor].reset_index(drop=True)    # reset index so 0..n

    return df

def combine_all_subjects(subjects, downsample_factor=DOWNSAMPLE_FACTOR):
    # combine all the subjects into one big dataframe
    # each subject has ~19k rows ish, so together that's over ~250k

    dfs = [subject_to_dataframe(subj, downsample_factor) for subj in subjects]
    full_df = pd.concat(dfs, ignore_index=True)

    return full_df

# ==================================================================== MAIN SCRIPT
def main():
    print("Downloading Dataset...")
    download_wesad()

    print("\nLoading subjects...")
    subjects = load_all_subjects(RAW_DATA_PATH)
    print(f" * Loaded {len(subjects)} subjects\n")

    # combine and process all subjects
    full_df = combine_all_subjects(subjects)
    print(f" + Combined DataFrame shape: {full_df.shape}")

    # NOTE: for debugging / quick check
    print("\n - Label counts:")
    print(full_df['label'].value_counts())

    # save to CSV for better modeling
    full_df.to_csv(PROCESSED_FEATURES_PATH, index=False)
    print(f"\n ! Processed features saved to processed_features.csv\n")

    extract_features()

    model_data()

if __name__ == "__main__":
    main()