import os

import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset.PatientDataset import HomogeneousPatientDataset
from environment_setup import get_configurations_dtype_string


def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector

def normalize_features(features):
    for ii in range(np.shape(features)[1]):
    #    radiomics[:, ii] = z_score_normalize(radiomics[:, ii])
        features[:, ii] = min_max_normalize(features[:, ii], 1)

def prepare_dataset():
    all_labels = []
    all_features = []
    dataset = HomogeneousPatientDataset()
    for idx in range(len(dataset)):
        graph, label = dataset[idx]
        # Average across all nodes to generate a single feature
        all_features.append(torch.mean(graph.x, dim=0).numpy())
        all_labels.append(label)
    numpy_features = np.stack(all_features)
    numpy_labels = np.stack(all_labels)
    return numpy_features, numpy_labels


def get_folds():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    num_folds = pickle.load(open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'rb'))
    print(f"Using a pre-defined {num_folds} fold split. Done for easy reproducibility.")
    train_indices = pickle.load(open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'rb'))
    val_indices = pickle.load(open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'rb'))
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'rb'))
    return train_indices, val_indices, test_indices


def train_model(numpy_features, numpy_labels, solver):
    logistic_regression = LogisticRegression(class_weight="balanced", max_iter=1000, penalty='l2', solver=solver)
    avg_accuracy, avg_roc = [], []
    for fold, (train_idx, _, test_idx) in enumerate(zip(*get_folds())):
        train_features = numpy_features[train_idx]
        test_features = numpy_features[test_idx]
        train_labels = numpy_labels[train_idx]
        test_labels = numpy_labels[test_idx]
        model = logistic_regression.fit(train_features, train_labels)
        predictions = model.predict_proba(test_features)[:, 1]
        roc_score = roc_auc_score(test_labels, predictions)
        predicted_label = model.predict(test_features)
        acc = np.sum((predicted_label == test_labels)) / predictions.shape[0]
        avg_accuracy.append(acc)
        avg_roc.append(roc_score)
    return np.mean(avg_roc), np.std(avg_roc), np.mean(avg_accuracy), np.std(avg_accuracy)


if __name__ == '__main__':
    numpy_features, numpy_labels = prepare_dataset()
    normalize_features(features=numpy_features)
    for solver in ['lbfgs', 'newton-cg', 'liblinear', 'saga']:
        roc, roc_std, acc, acc_std = train_model(numpy_features=numpy_features, numpy_labels=numpy_labels, solver=solver)
        evaluation_results = f'Solver {solver}, ROC: {roc:.3f} ± {roc_std:.3f} and accuracy {acc:.3f} ± {acc_std:.3f}'
        print(evaluation_results)