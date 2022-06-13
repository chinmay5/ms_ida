import os
import pickle
import torch
from sklearn.model_selection import train_test_split

from dataset.PatientDataset import HeterogeneousPatientDataset, HomogeneousPatientDataset
from environment_setup import get_configurations_dtype_string


def generate_train_val_test_indices_for_single_split(dataset):
    train_indices, test_indices, _, _ = train_test_split(torch.arange(len(dataset)),
                                                         dataset.y, test_size=0.11, stratify=dataset.y, random_state=42)
    train_and_val_dataset = [dataset[x.item()] for x in train_indices]
    train_and_val_labels = [dataset[x.item()][1] for x in train_indices]
    train_indices, val_indices, _, _ = train_test_split(torch.arange(len(train_and_val_dataset)),
                                                        train_and_val_labels, test_size=0.125,
                                                        stratify=train_and_val_labels, random_state=42)
    return train_indices, val_indices, test_indices


def save_graph_subsets_based_on_splits(dataset, train_idx, val_idx, test_idx, filename_suffix):
    train_dataset = [dataset[x.item()] for x in train_idx]
    val_dataset = [dataset[x.item()] for x in val_idx]
    test_dataset = [dataset[x.item()] for x in test_idx]
    temp_folder_path = get_configurations_dtype_string(section='SETUP', key='TEMP_FOLDER_PATH')
    os.makedirs(temp_folder_path, exist_ok=True)
    pickle.dump(train_dataset, open(os.path.join(temp_folder_path, f'train_set_{filename_suffix}.pkl'), 'wb'))
    pickle.dump(val_dataset, open(os.path.join(temp_folder_path, f'val_set_{filename_suffix}.pkl'), 'wb'))
    pickle.dump(test_dataset, open(os.path.join(temp_folder_path, f'test_set_{filename_suffix}.pkl'), 'wb'))


if __name__ == '__main__':
    data_het = HeterogeneousPatientDataset()
    data_hom = HomogeneousPatientDataset()
    train_indices, val_indices, test_indices = generate_train_val_test_indices_for_single_split(dataset=data_hom)
    save_graph_subsets_based_on_splits(dataset=data_het, train_idx=train_indices, val_idx=val_indices,
                                       test_idx=test_indices, filename_suffix='het')
    save_graph_subsets_based_on_splits(dataset=data_hom, train_idx=train_indices, val_idx=val_indices,
                                       test_idx=test_indices, filename_suffix='hom')
