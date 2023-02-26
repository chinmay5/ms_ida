from collections import defaultdict, Counter

import os
import pickle
import torch

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string
from utils.training_utils import decide_graph_category_based_on_size


def load_split(split_idx=6):
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    num_folds = pickle.load(open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'rb'))
    print(f"Using a pre-defined {num_folds} fold split. Done for easy reproducibility.")
    all_train_indices = pickle.load(open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'rb'))
    all_val_indices = pickle.load(open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'rb'))
    all_test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'rb'))
    # Now, we can go ahead and select the labels based on the split_idx
    train_indices, val_indices, test_indices = all_train_indices[split_idx], all_val_indices[split_idx], \
                                               all_test_indices[split_idx]
    return train_indices, val_indices, test_indices


def print_distribution_stats_for_dataset(train_dataset, val_dataset, test_dataset):
    graph_size_2_labels_list_dict_train = build_graph_size_2_labels_dict(train_dataset)
    graph_size_2_labels_list_dict_val = build_graph_size_2_labels_dict(val_dataset)
    graph_size_2_labels_list_dict_test = build_graph_size_2_labels_dict(test_dataset)
    for graph_size in graph_size_2_labels_list_dict_train.keys():
        print(
            f"{graph_size} has the distribution \nTRAIN: {Counter(graph_size_2_labels_list_dict_train[graph_size])}\n"
            f"VALID: {Counter(graph_size_2_labels_list_dict_val[graph_size])} \nTEST:"
            f"{Counter(graph_size_2_labels_list_dict_test[graph_size])}\n\n")


def build_graph_size_2_labels_dict(dataset_to_use):
    graph_len_2_labels = defaultdict(list)
    for idx in range(len(dataset_to_use)):
        graph, label = dataset_to_use[idx]
        graph_size_categ = decide_graph_category_based_on_size(graph.x.size(0))
        graph_len_2_labels[graph_size_categ].append(label)
    return graph_len_2_labels


def get_statistics(train_val_test_indices):
    train_indices = train_val_test_indices[0]
    val_indices = train_val_test_indices[1]
    test_indices = train_val_test_indices[2]
    dataset = get_dataset()
    # Now select the dataset splits based on indices.
    train_dataset = [dataset[idx.item()] for idx in train_indices]
    test_dataset = [dataset[idx.item()] for idx in test_indices]
    val_dataset = [dataset[idx.item()] for idx in val_indices]

    # We also need to obtain class weights to ensure we do not have data imbalance issues.
    pos_samples = sum([sample[1] for sample in train_dataset])
    neg_samples = len(train_dataset) - pos_samples
    if pos_samples > neg_samples:
        class_balance_weights = torch.as_tensor([pos_samples / neg_samples, 1])
    else:
        class_balance_weights = torch.as_tensor([1, neg_samples / pos_samples])
    # print(f"The class balance weight is {class_balance_weights}")
    # Let us print some statistics about the dataset to see if there is a distribution shift
    print_distribution_stats_for_dataset(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)


if __name__ == '__main__':
    for split_idx in range(10):
        train_val_test_indices = load_split(split_idx=split_idx)
        get_statistics(train_val_test_indices=train_val_test_indices)
