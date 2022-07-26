from collections import defaultdict

import os

import pickle

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from dataset.PatientDataset import KNNPatientDataset
from environment_setup import get_configurations_dtype_string
from utils.eval_utils import decide_graph_category_based_on_size, plot_avg_of_dictionary, \
    print_custom_avg_of_dictionary, pretty_print_avg_dictionary


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
    all_graph_categ = []
    dataset = KNNPatientDataset()
    for idx in range(len(dataset)):
        graph, label = dataset[idx]
        # Average across all nodes to generate a single feature
        all_features.append(torch.mean(graph.x, dim=0).numpy())
        all_labels.append(label)
        all_graph_categ.append(decide_graph_category_based_on_size(graph.x.size(0)))
    numpy_features = np.stack(all_features)
    numpy_labels = np.stack(all_labels)
    numpy_graph_categ = np.stack(all_graph_categ)
    return numpy_features, numpy_labels, numpy_graph_categ


def get_folds():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    num_folds = pickle.load(open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'rb'))
    print(f"Using a pre-defined {num_folds} fold split. Done for easy reproducibility.")
    train_indices = pickle.load(open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'rb'))
    val_indices = pickle.load(open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'rb'))
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'rb'))
    return train_indices, val_indices, test_indices


def compute_numpy_label_wise_acc(test_labels, predicted_label):
    zero_indices = np.where(test_labels == 0)[0]
    ones_indices = np.where(test_labels == 1)[0]
    zero_acc = (predicted_label[zero_indices] == test_labels[zero_indices]).sum().item() / zero_indices.shape[0]
    ones_acc = (predicted_label[ones_indices] == test_labels[ones_indices]).sum().item() / ones_indices.shape[0]
    return zero_acc, ones_acc


def train_model(numpy_features, numpy_labels, solver, numpy_graph_categ, split_acc_based_on_labels):
    logistic_regression = LogisticRegression(class_weight="balanced", max_iter=1000, penalty='l2', solver=solver)
    avg_accuracy, avg_roc = [], []
    graph_type_acc_dict, graph_type_roc_dict = defaultdict(list), defaultdict(list)
    label_wise_acc_dict = defaultdict(list)
    # Since this is supposed to be a standalone script, let us just save the results in the current folder itself.
    test_images_dir = os.getcwd()
    for fold, (train_idx, _, test_idx) in enumerate(zip(*get_folds())):
        size_cm_dict = defaultdict(list)
        train_features = numpy_features[train_idx]
        test_features = numpy_features[test_idx]
        train_labels = numpy_labels[train_idx]
        test_labels = numpy_labels[test_idx]
        # We would need to know about the graph_categories for only the test labels
        for idx in test_idx:
            size_cm_dict[numpy_graph_categ[idx]].append((numpy_features[idx, :], numpy_labels[idx]))
        model = logistic_regression.fit(train_features, train_labels)
        predictions = model.predict_proba(test_features)[:, 1]
        roc_score = roc_auc_score(test_labels, predictions)
        predicted_label = model.predict(test_features)
        acc = np.sum((predicted_label == test_labels)) / predictions.shape[0]
        avg_accuracy.append(acc)
        avg_roc.append(roc_score)

        for graph_size, test_features_and_labels in size_cm_dict.items():
            test_features = np.stack([x[0] for x in test_features_and_labels])
            test_labels = np.stack([x[1] for x in test_features_and_labels])
            predictions = model.predict_proba(test_features)[:, 1]
            predicted_label = model.predict(test_features)
            if split_acc_based_on_labels:
                zero_acc, ones_acc = compute_numpy_label_wise_acc(test_labels, predicted_label)
                label_wise_acc_dict[graph_size].append((zero_acc, ones_acc))
            acc = np.sum((predicted_label == test_labels)) / predictions.shape[0]
            roc_score = roc_auc_score(test_labels, predictions)
            graph_type_acc_dict[graph_size].append(acc)
            graph_type_roc_dict[graph_size].append(roc_score)

    plot_avg_of_dictionary(input_dict=graph_type_roc_dict, y_label='roc', filename=f'{model}_roc_avg_all_folds',
                           output_dir=test_images_dir, color='m')
    plot_avg_of_dictionary(input_dict=graph_type_acc_dict, y_label='acc', filename=f'{model}_acc_avg_all_folds',
                           output_dir=test_images_dir, color='m')
    if split_acc_based_on_labels:
        print_custom_avg_of_dictionary(input_dict=label_wise_acc_dict)
    pretty_print_avg_dictionary(input_dict=graph_type_roc_dict)
    return np.mean(avg_roc), np.std(avg_roc), np.mean(avg_accuracy), np.std(avg_accuracy)


if __name__ == '__main__':
    numpy_features, numpy_labels, numpy_graph_categ = prepare_dataset()
    normalize_features(features=numpy_features)
    for solver in ['lbfgs', 'newton-cg', 'liblinear', 'saga']:
        roc, roc_std, acc, acc_std = train_model(numpy_features=numpy_features, numpy_labels=numpy_labels,
                                                 numpy_graph_categ=numpy_graph_categ, solver=solver,
                                                 split_acc_based_on_labels=True)
        evaluation_results = f'Solver {solver}, ROC: {roc:.3f} ± {roc_std:.3f} and accuracy {acc:.3f} ± {acc_std:.3f}'
        print(evaluation_results)
