# Stress-Detection
Rather than focusing only on prediction accuracy, this work emphasizes **understanding the patterns in the data** — specifically, whether stress-related signals are actually separable from normal physiological behavior.

Using the WESAD dataset, we analyze signals such as ECG (heart activity), EDA (skin conductance), and temperature to evaluate how well different machine learning models can detect stress.


## Key Idea
Even if a model appears "accurate," it may still fail at the task that matters: detecting stress.

This project investigates:
- How we represent physiological data
- Whether stress patterns are distinguishable from baseline
- How model behavior changes when stress is treated as a higher-risk condition


## Project Pipeline

Estimated time for execution: ~5 minutes

### 1. Data Processing
```
python main.py
```
- Loads raw .pkl files from the dataset
- Filters data to baseline (1) and stress (2) labels
- Downsamples signals for efficiency
- Outputs: processed_features.csv

### 2. Feature Extraction
```
python featureExtraction.py
```
- Segments data into fixed-size time windows
- Computes statistical features (mean, std, min, max)
- Outputs: windowed_features.csv

### 3. Modeling and Analysis
```
python models.py
```
- Trains and evaluates:
  - Support Vector Machines (linear and RBF)
  - K-Nearest Neighbors
  - Naïve Bayes
- Compares different feature sets (ECG, EDA, combined)
- Outputs classification metrics and PCA visualizations

### All Libraries and Imports used:
```
import os
import pickle
import warnings

import pandas as pd
from src.parse.load import load_all_subjects

from sklearn.naive_bayes import GaussianNB            
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
```

## Dataset
### WESAD (Wearable Stress and Affect Detection Dataset)
- Multimodal physiological signals from wearable devices
- Includes ECG, EDA, temperature, and motion data
- Contains labeled conditions such as baseline and stress

Download from:
```
https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset?utm_source=chatgpt.com
```
Place the dataset inside the `/data` folder before running the project.


## Summary of Findings
- Stress and baseline signals show significant overlap
- Models achieve moderate accuracy (~65%) but fail to detect stress reliably
- Most models default to predicting the baseline state
- Stress detection recall is very low, indicating missed stress cases

### Key takeaway:
- The limitation is not just the models, but also how the data is represented.
- Simple statistical features do not fully capture the complexity of stress responses.


### Future Work
- Explore subject-specific models
- Incorporate temporal and frequency-based features (e.g., HRV)
- Improve feature representation for better separability


## Authors
- Amelia McCormack
- Lillian Malik