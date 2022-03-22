import numpy as np
import scipy.io as sio
import pandas as pd
import matplotlib.pyplot as plt
import os
from data.preparator import Preparator
from sklearn.model_selection import train_test_split

# Train & pickle MDM classifier
import pyriemann

# Train and pickle SVM w/Riemannian Kernel
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

DIRNAME = 'data/prepared_subjects'


def run_MDM(X_train, X_test, y_train, y_test):
    # transform X_train, X_test into covariance matrices
    cov = Covariances('oas').fit_transform(np.swapaxes(X_train, 1, 2))
    test_cov = Covariances('oas').fit_transform(np.swapaxes(X_test, 1, 2))

    mdm = pyriemann.classification.MDM('riemann')

    print(f"Fitting MDM model...")
    fitted = mdm.fit(cov, y_train)
    result = fitted.score(test_cov, y_test)
    return result


def run_SVM(X, y):
    # build your pipeline
    covest = Covariances('oas')
    ts = TangentSpace()
    svc = SVC(kernel='linear')
    print("Fitting SVM Model ...")
    clf = make_pipeline(covest, ts, svc)
    # cross validation
    accuracy = cross_val_score(clf, X, y, cv=5)
    return accuracy.mean()


def run_riemannian(filename):
    with np.load(f"{DIRNAME}/{filename}") as f:
        data = f["EEG"]
        labels = f["labels"][:, 1]

    print(f"Subject {filename} EEG Shape: {data.shape}")

    # ##################### MDM ############################
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42)

    mdm_accuracy = run_MDM(X_train, X_test, y_train, y_test)
    print(f"MDM accuracy: {mdm_accuracy}")

    ##################### SVM ############################
    # load your data
    # your EEG data, in format Ntrials x Nchannels X Nsamples
    X = np.swapaxes(data, 1, 2)
    y = labels
    svm_accuracy = run_SVM(X, y)
    print(f"SVM accuracy: {svm_accuracy}")

    return mdm_accuracy, svm_accuracy


def main():
    cols = ['subject', 'mdm', 'svm']
    results = pd.DataFrame(columns=cols)

    filenames = sorted(os.listdir(DIRNAME))
    for f in filenames:
        print(f"Currently training on subject {f}")
        mdm_accuracy, svm_accuracy = run_riemannian(f)

        # save results
        subject = f.split('.')[0]
        subject_results = pd.DataFrame(
            [[subject, mdm_accuracy, svm_accuracy]], columns=cols)
        results = pd.concat([results, subject_results])

    results.to_csv('classification_results.csv')


main()
