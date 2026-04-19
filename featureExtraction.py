import pandas as pd

# ==================================================================== LOAD DATA

print("Loading processed feature data...")
df = pd.read_csv("processed_features.csv")

print(f" * Loaded dataframe shape: {df.shape}")
print("\n - Label distribution:")
print(df['label'].value_counts())

# ==================================================================== FEATURE WINDOWING

windowSize = 50
features = []
total_subjects = df['subject_id'].nunique()
subject_counter = 0

print("\nCreating sliding windows...")

for subject, sub_df in df.groupby('subject_id'):

    subject_counter += 1
    print(f"\n * Processing subject {subject} ({subject_counter}/{total_subjects})")

    for i in range(0, len(sub_df) - windowSize, windowSize):
        window = sub_df.iloc[i:i + windowSize]

        feature = {
            # ---------------- ECG FEATURES
            'ECG_mean': window['ECG'].mean(),
            'ECG_std': window['ECG'].std(),
            'ECG_min': window['ECG'].min(),
            'ECG_max': window['ECG'].max(),

            # ---------------- EDA FEATURES
            'EDA_mean': window['EDA'].mean(),
            'EDA_std': window['EDA'].std(),
            'EDA_min': window['EDA'].min(),
            'EDA_max': window['EDA'].max(),

            # ---------------- TEMP FEATURES
            'TEMP_mean': window['TEMP'].mean(),
            'TEMP_std': window['TEMP'].std(),

            'label': window['label'].mode()[0],     # LABELING
            'subject_id': subject
        }
        features.append(feature)

# ==================================================================== FINALIZE DATASET
featureDF = pd.DataFrame(features)

featureDF.to_csv("windowed_features.csv", index=False)
print("\n! Saved windowed dataset to windowed_features.csv")

print("Total windows:", len(featureDF))
