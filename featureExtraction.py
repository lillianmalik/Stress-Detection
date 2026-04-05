import pandas as pd

df = pd.read_csv("processed_features.csv")

windowSize = 50
features = []

for subject, sub_df in df.groupby('subject_id'):
    for i in range(0, len(sub_df) - windowSize, windowSize):
        window = sub_df.iloc[i:i + windowSize]

        feature = {
            'ECG_mean': window['ECG'].mean(),
            'ECG_std': window['ECG'].std(),
            'ECG_min': window['ECG'].min(),
            'ECG_max': window['ECG'].max(),

            'EDA_mean': window['EDA'].mean(),
            'EDA_std': window['EDA'].std(),
            'EDA_min': window['EDA'].min(),
            'EDA_max': window['EDA'].max(),

            'TEMP_mean': window['TEMP'].mean(),
            'TEMP_std': window['TEMP'].std(),

            'label': window['label'].mode()[0],
            'subject_id': subject
        }
        features.append(feature)
featureDF = pd.DataFrame(features)
featureDF.to_csv("windowed_features.csv", index=False)

print("Total windows:", len(featureDF))

