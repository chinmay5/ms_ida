from collections import Counter

import pickle

import os
import pandas as pd
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Dataset

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_int, \
    get_configurations_dtype_boolean


class HomogeneousPatientDataset(Dataset):
    def __init__(self, transform=None):
        super(HomogeneousPatientDataset, self).__init__()
        annotated_data_csv_location = get_configurations_dtype_string(section='SETUP',
                                                                      key='ANNOTATED_DATA_CSV_LOCATION')
        annotated_data = pd.read_csv(annotated_data_csv_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = annotated_data.loc[:, 'New_Lesions_1y_Label']
        self.graph_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_HETERO_DATASET_ROOT_FOLDER')
        self.remove_knn_edges = get_configurations_dtype_boolean(section='TRAINING', key='REMOVE_KNN_EDGES', default_value=False)
        self.transform = transform

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        graph_name = f"{self.patient_list[item]}.pt"
        graph_label = self.y[item].item()
        heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))
        if self.remove_knn_edges:
            del heterogeneous_graph[('lesion', 'NN', 'lesion')]
        # Let us make it homogeneous
        homogeneous_graph = self.convert_to_homogeneous_graph(heterogeneous_graph)
        # Column normalize the features
        if self.transform is not None:
            homogeneous_graph = self.transform(homogeneous_graph)
        return homogeneous_graph, graph_label

    def convert_to_homogeneous_graph(self, heterogeneous_graph):
        homogeneous_graph = heterogeneous_graph.to_homogeneous()
        # Let us check if the graph is an isolated one.
        # In this case, the edge indices are removed.
        # We add it explicitly to ensure collate function does not fail.
        if not 'edge_index' in homogeneous_graph:
            homogeneous_graph['edge_index'] = torch.tensor([[], []], dtype=torch.long)
        # This graph is missing the edge_type indicating that it is an isolated graph
        if not 'edge_types' in heterogeneous_graph:
            homogeneous_graph['edge_type'] = torch.tensor([], dtype=torch.long)
        return homogeneous_graph

    @property
    def num_graphs(self):
        return len(self.y)


class HeterogeneousPatientDataset(Dataset):
    def __init__(self, transform=None):
        super(HeterogeneousPatientDataset, self).__init__()
        annotated_data_csv_location = get_configurations_dtype_string(section='SETUP',
                                                                      key='ANNOTATED_DATA_CSV_LOCATION')
        annotated_data = pd.read_csv(annotated_data_csv_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = torch.as_tensor(annotated_data.loc[:, 'New_Lesions_1y_Label'], dtype=torch.long)
        self.graph_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_HETERO_DATASET_ROOT_FOLDER')
        self.remove_knn_edges = get_configurations_dtype_boolean(section='TRAINING', key='REMOVE_KNN_EDGES', default_value=False)
        self.transform = transform

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        graph_name = f"{self.patient_list[item]}.pt"
        graph_label = self.y[item].item()
        heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))
        if self.transform is not None:
            heterogeneous_graph = self.transform(heterogeneous_graph)
        heterogeneous_graph = self.handle_isolated_edges(heterogeneous_graph)
        # This step done later since it is much more convenient.
        # Remove the edges once all bases related to it are covered.
        if self.remove_knn_edges:
            del heterogeneous_graph[('lesion', 'NN', 'lesion')]
        return heterogeneous_graph, graph_label

    def __repr__(self):
        return f"Heterogeneous graph with {self.y.size(0)} labels"

    def handle_isolated_edges(self, heterogeneous_graph):
        # We next handle if this was an isolated graph itself.
        # In such a scenario, we delete the edge_index
        # and allow it to become a heterogeneous graph.
        if 'edge_index' in heterogeneous_graph['lesion']:
            del heterogeneous_graph['lesion']['edge_index']
            heterogeneous_graph[('lesion', 'LesionLocation', 'lesion')].edge_index = torch.tensor([[], []],
                                                                                                  dtype=torch.long)
            heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_index = torch.tensor([[], []], dtype=torch.long)
        # We also handle cases where one kind of edge is missing.
        if ('lesion', 'LesionLocation', 'lesion') not in heterogeneous_graph.edge_index_dict:
            heterogeneous_graph[('lesion', 'LesionLocation', 'lesion')].edge_index = torch.tensor([[], []],
                                                                                                  dtype=torch.long)
        if ('lesion', 'NN', 'lesion') not in heterogeneous_graph.edge_index_dict:
            heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_index = torch.tensor([[], []], dtype=torch.long)
        return heterogeneous_graph

    @property
    def num_graphs(self):
        return len(self.y)


def separate_large_and_small_graphs():
    largeness_threshold = get_configurations_dtype_int(section='SETUP', key='LARGENESS_THRESHOLD')
    large_graphs_filename = get_configurations_dtype_string(section='SETUP', key='LARGE_GRAPHS_FILENAME')
    small_graphs_filename = get_configurations_dtype_string(section='SETUP', key='SMALL_GRAPHS_FILENAME')

    heterogeneous_dataset = HeterogeneousPatientDataset()
    small_patients_list, large_patients_list = [], []
    for idx in range(len(heterogeneous_dataset)):
        candidate_graph = heterogeneous_dataset[idx]
        if candidate_graph[0].x_dict['lesion'].shape[0] > largeness_threshold:
            large_patients_list.append(candidate_graph)
        else:
            small_patients_list.append(candidate_graph)
    pickle.dump(large_patients_list, open(large_graphs_filename, 'wb'))
    pickle.dump(small_patients_list, open(small_graphs_filename, 'wb'))


if __name__ == '__main__':
    transform = T.NormalizeFeatures()
    dataset = HeterogeneousPatientDataset(transform=transform)
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=4, shuffle=False)
    print(dataset[0])
    print(dataset.num_graphs)
    all_labels = []
    for graph, label in dataloader:
        all_labels.extend(label.numpy().tolist())
    print(Counter(all_labels))

