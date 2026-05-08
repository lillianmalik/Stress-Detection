# models.py
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from src.util.paths import WINDOWED_FEATURES_PATH

def model_data():
    # ==================================================================== LOAD DATA
    print("Loading windowed dataset...")
    df = pd.read_csv(WINDOWED_FEATURES_PATH)

    print(f" * Loaded shape: {df.shape}")

    train = df[df['subject_id'] != 'S2']
    test = df[df['subject_id'] == 'S2']

    y_train = train['label']
    y_test = test['label']

    if test.shape[0] == 0:
        raise ValueError("Test subject S2 not found in dataset")    # safety check

    print(f" * Train shape: {train.shape}")
    print(f" * Test subject: S2 ({test.shape})")

    # ==================================================================== TRAINING & MODELS

    print("\nStarting model training pipeline...")

    # ---------------- ECG ONLY
    xECG_Train =  train[['ECG_mean', 'ECG_std']]
    xECG_Test = test[['ECG_mean', 'ECG_std']]
    modelECG =  SVC(kernel="linear")
    modelECG.fit(xECG_Train, y_train)

    yECG_Pred = modelECG.predict(xECG_Test)
    print("\n--- ECG Only ---")
    print("Accuracy:", modelECG.score(xECG_Test, y_test))
    print(confusion_matrix(y_test, yECG_Pred, labels=[1, 2]))
    print(classification_report(y_test, yECG_Pred, labels=[1, 2], zero_division=0))


    # ---------------- EDA ONLY
    xEDA_Train =  train[['EDA_mean', 'EDA_std']]
    xEDA_Test = test[['EDA_mean', 'EDA_std']]
    modelEDA =  SVC(kernel="linear")
    modelEDA.fit(xEDA_Train, y_train)

    yEDA_Pred = modelEDA.predict(xEDA_Test)
    print("\n--- EDA Only ---")
    print("Accuracy:", modelEDA.score(xEDA_Test, y_test))
    print(confusion_matrix(y_test, yEDA_Pred, labels=[1, 2]))
    print(classification_report(y_test, yEDA_Pred, labels=[1, 2], zero_division=0))


    # ---------------- BALANCED EDA
    xEDA_Train =  train[['EDA_mean', 'EDA_std']]
    xEDA_Test = test[['EDA_mean', 'EDA_std']]
    modelBalanced =  SVC(kernel="linear", class_weight="balanced")
    modelBalanced.fit(xEDA_Train, y_train)

    yBalanced_Pred = modelBalanced.predict(xEDA_Test)
    print("\n--- BALANCED EDA Only ---")
    print("Accuracy:", modelBalanced.score(xEDA_Test, y_test))
    print(confusion_matrix(y_test, yBalanced_Pred, labels=[1, 2]))
    print(classification_report(y_test, yBalanced_Pred, labels=[1, 2], zero_division=0))


    # ---------------- Combined
    xALL_Train =  train[['ECG_mean', 'ECG_std','EDA_mean', 'EDA_std', 'TEMP_mean']]
    xALL_Test =  test[['ECG_mean', 'ECG_std','EDA_mean', 'EDA_std', 'TEMP_mean']]

    modelALL =  SVC(kernel="linear")
    modelALL.fit(xALL_Train, y_train)

    yALL_Pred = modelALL.predict(xALL_Test)
    print("\n--- Combined Features ---")
    print("Accuracy:", modelALL.score(xALL_Test, y_test))
    print(confusion_matrix(y_test, yALL_Pred, labels=[1, 2]))
    print(classification_report(y_test, yALL_Pred, labels=[1, 2], zero_division=0))


    # ---------------- RBF SVM
    rbf_model = SVC(kernel='rbf')
    rbf_model.fit(xALL_Train, y_train)

    yPredRBF = rbf_model.predict(xALL_Test)
    print("\n--- SVM RBF ---")
    print("Accuracy:", rbf_model.score(xALL_Test, y_test))
    print(confusion_matrix(y_test, yPredRBF, labels=[1, 2]))
    print(classification_report(y_test, yPredRBF, labels=[1, 2], zero_division=0))


    # ---------------- KNN
    knn = KNeighborsClassifier()
    knn.fit(xALL_Train, y_train)

    yPredKNN = knn.predict(xALL_Test)
    print("\n--- KNN ---")
    print("Accuracy:", knn.score(xALL_Test, y_test))
    print(confusion_matrix(y_test, yPredKNN, labels=[1, 2]))
    print(classification_report(y_test, yPredKNN, labels=[1, 2], zero_division=0))


    # ---------------- Naive Bayes
    nb = GaussianNB()
    nb.fit(xALL_Train, y_train)

    yPrednb = nb.predict(xALL_Test)
    print("\n--- Naive Bayes ---")
    print("Accuracy:", nb.score(xALL_Test, y_test))
    print(confusion_matrix(y_test, yPrednb, labels=[1, 2]))
    print(classification_report(y_test, yPrednb, labels=[1, 2], zero_division=0))

    # ==================================================================== PCA VISUALIZATION
    print("\nGenerating PCA visualization...")

    # ---------------- Combined Individual Variability
    x = df[['ECG_mean', 'ECG_std', 'EDA_mean', 'EDA_std', 'TEMP_mean']]
    pca = PCA(n_components=2)
    xPCA = pca.fit_transform(x)

    subject = df['subject_id'].astype('category').cat.codes

    plt.figure(figsize=(9, 7))

    # ---------------- Baseline (circle)
    baseline = df['label'] == 1
    plt.scatter(
        xPCA[baseline, 0],
        xPCA[baseline, 1],
        c=subject[baseline],
        cmap='tab20',
        marker='o',
        s=15,
        alpha=0.7,
        label='Baseline'
    )

    # ---------------- Stress
    stress = df['label'] == 2
    plt.scatter(
        xPCA[stress, 0],
        xPCA[stress, 1],
        c=subject[stress],
        cmap='tab20',
        marker='x',
        s=25,
        alpha=0.9,
        label='Stress'
    )

    plt.title("PCA: Subject Clusters with Baseline vs. Stress")

    plt.xlabel(f"Principal Component 1")
    plt.ylabel(f"Principal Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\n! PCA visualization complete\n")