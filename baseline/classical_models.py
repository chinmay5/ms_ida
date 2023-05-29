from collections import defaultdict

import os

import pickle

import numpy as np
import torch
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR
from utils.training_utils import plot_avg_of_dictionary, print_custom_avg_of_dictionary, pretty_print_avg_dictionary, \
    decide_graph_category_based_on_size, k_fold, normalize_features


class DataFetcher(object):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.start_idx = 0

    def get_features_for_indices(self, indices):
        features = []
        size_tuple = []
        for idx in indices:
            graph_features = self.dataset[idx][0].x
            features.append(graph_features)
            end_idx = self.start_idx + graph_features.shape[0]
            size_tuple.append((self.start_idx, end_idx))
            self.start_idx = end_idx
        # Reset the start_idx
        self.start_idx = 0
        return torch.cat(features), size_tuple


fetcher = DataFetcher(dataset=get_dataset())


def prepare_dataset():
    all_labels = []
    all_features = []
    all_graph_categ = []
    all_regr_target = []
    dataset = get_dataset()
    print(dataset)
    for idx in range(len(dataset)):
        graph_orig, _, label = dataset[idx]
        # Average across all nodes to generate a single feature
        # all_features.append(torch.mean(graph_orig.x, dim=0).numpy())
        all_features.append(torch.sum(graph_orig.x, dim=0).numpy())
        all_labels.append(label)
        all_graph_categ.append(decide_graph_category_based_on_size(graph_orig.x.size(0)))
        all_regr_target.append(graph_orig.graph_vol)
    # Now we stack the results together.
    numpy_features = np.stack(all_features)
    numpy_labels = np.stack(all_labels)
    numpy_graph_categ = np.stack(all_graph_categ)
    numpy_regr_labels = np.stack(all_regr_target)
    return numpy_features, numpy_labels, numpy_graph_categ, numpy_regr_labels


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


def get_concatenated_features(train_ids, test_ids, num_layers, hidden_dim, target_epoch, fold, only_ssl):
    raw_train_features, train_size_tuple = fetcher.get_features_for_indices(train_ids.numpy())
    raw_test_features, test_size_tuple = fetcher.get_features_for_indices(test_ids.numpy())
    # We are including the ssl features here
    log_dir = os.path.join(os.path.join(PROJECT_ROOT_DIR,
                                        get_configurations_dtype_string(section='TRAINING', key='LOG_DIR')),
                           f"_ssl_layers_{num_layers}_hidden_dim_{hidden_dim}")
    folder_base = os.path.join(log_dir, 'ssl_features', str(target_epoch))
    train_ssl_features = np.load(os.path.join(folder_base, f'train_ssl_{fold}.npy'))
    test_ssl_features = np.load(os.path.join(folder_base, f'test_ssl_{fold}.npy'))
    if not only_ssl:
        train_features = np.concatenate((raw_train_features, train_ssl_features), axis=1)
        test_features = np.concatenate((raw_test_features, test_ssl_features), axis=1)
        # Now, we need to take a mean of the features belonging to the same patient
    else:
        train_features = train_ssl_features
        test_features = test_ssl_features
    train_features = np.stack([np.mean(train_features[start: end], axis=0) for start, end in train_size_tuple])
    test_features = np.stack([np.mean(test_features[start: end], axis=0) for start, end in test_size_tuple])
    return train_features, test_features


def train_classification_model(numpy_features, numpy_labels, numpy_graph_categ, split_acc_based_on_labels,
                               model_type, use_ssl=False, only_ssl=False,
                               num_layers=-1, hidden_dim=-1, target_epoch=-1):
    assert model_type in ['lr', 'rf', 'svm'], "Invalid model choice"
    if model_type == 'lr':
        model = LogisticRegression(class_weight="balanced", penalty='l2', solver='liblinear', random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(class_weight="balanced", random_state=42)
    else:
        model = svm.SVC(class_weight="balanced", random_state=42, probability=True, gamma='auto')
    print(f"Model is {model}")
    avg_accuracy, avg_roc = [], []
    graph_type_acc_dict, graph_type_roc_dict = defaultdict(list), defaultdict(list)
    label_wise_acc_dict = defaultdict(list)
    # Since this is supposed to be a standalone script, let us just save the results in the current folder itself.
    test_images_dir = os.getcwd()
    folds = 10
    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(get_dataset(), folds))):
        # Add the ssl features
        if use_ssl:
            train_features, test_features = get_concatenated_features(train_ids=train_idx, test_ids=test_idx,
                                                                      num_layers=num_layers, hidden_dim=hidden_dim,
                                                                      target_epoch=target_epoch, fold=fold,
                                                                      only_ssl=only_ssl)
        else:
            train_features = numpy_features[train_idx]
            test_features = numpy_features[test_idx]
        size_cm_dict = defaultdict(list)
        train_labels = numpy_labels[train_idx]
        test_labels = numpy_labels[test_idx]

        # We would need to know about the graph_categories for only the test labels
        for mapped_idx, idx in enumerate(test_idx):
            size_cm_dict[numpy_graph_categ[idx]].append((test_features[mapped_idx, :], numpy_labels[idx]))
        # Just to be sure on the shape mismatch
        train_labels, test_labels = train_labels.squeeze(), test_labels.squeeze()
        model = model.fit(train_features, train_labels)
        predictions = model.predict_proba(test_features)[:, 1]
        roc_score = roc_auc_score(test_labels, predictions, average='weighted')
        predicted_label = model.predict(test_features)
        acc = np.sum((predicted_label == test_labels)) / predictions.shape[0]
        avg_accuracy.append(acc)
        avg_roc.append(roc_score)

        for graph_size, test_features_and_labels in size_cm_dict.items():
            test_features = np.stack([x[0] for x in test_features_and_labels])
            test_labels = np.stack([x[1] for x in test_features_and_labels]).squeeze()
            predictions = model.predict_proba(test_features)[:, 1]
            predicted_label = model.predict(test_features)
            if split_acc_based_on_labels:
                try:
                    zero_acc, ones_acc = compute_numpy_label_wise_acc(test_labels, predicted_label)
                    label_wise_acc_dict[graph_size].append((zero_acc, ones_acc))
                except ZeroDivisionError as e:
                    continue
            acc = np.sum((predicted_label == test_labels)) / predictions.shape[0]
            roc_score = roc_auc_score(test_labels, predictions, average='weighted')
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


def train_regression_model(numpy_features, numpy_labels, is_logistic_regr=True):
    if is_logistic_regr:
        model = LinearRegression()
    else:
        model = RandomForestRegressor(random_state=42, criterion='absolute_error')
    print(f"Model is {model}")
    avg_error = []
    folds = 10
    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(get_dataset(), folds))):
        size_cm_dict = defaultdict(list)
        train_features = numpy_features[train_idx]
        test_features = numpy_features[test_idx]
        train_labels = numpy_labels[train_idx]
        test_labels = numpy_labels[test_idx]
        model = model.fit(train_features, train_labels)
        predictions = model.predict(test_features)
        error = np.linalg.norm((predictions - test_labels), ord=1) / test_labels.shape[0]
        avg_error.append(error)
    return np.mean(avg_error), np.std(avg_error)


if __name__ == '__main__':
    numpy_features, numpy_labels, numpy_graph_categ, numpy_regr_labels = prepare_dataset()
    # We are already normalizing features while creating the dataset.
    normalize_features(features=numpy_features)
    # for solver in ['lbfgs', 'newton-cg', 'liblinear', 'saga']:
    for model_type in ['rf', 'svm', 'lr']:
        roc, roc_std, acc, acc_std = train_classification_model(numpy_features=numpy_features,
                                                                numpy_labels=numpy_labels,
                                                                numpy_graph_categ=numpy_graph_categ,
                                                                split_acc_based_on_labels=True,
                                                                model_type=model_type,
                                                                use_ssl=False,
                                                                only_ssl=False,
                                                                hidden_dim=64,
                                                                num_layers=2,
                                                                target_epoch=199)
        evaluation_results = f'ROC: {roc:.3f} ± {roc_std:.3f} and accuracy {acc:.3f} ± {acc_std:.3f}'
        print(evaluation_results)

        # Same for the volume regression
        # if numpy_regr_labels is not None:
        #     error_mean, error_std = train_regression_model(numpy_features=numpy_features,
        #                                                    numpy_labels=numpy_regr_labels,
        #                                                    is_logistic_regr=False
        #                                                    # numpy_graph_categ=numpy_graph_categ,
        #                                                    # solver=solver,
        #                                                    # split_acc_based_on_labels=True
        #                                                    )
        #     evaluation_results = f'Error: {error_mean:.3f} ± {error_std:.3f}'
        #     print(evaluation_results)
