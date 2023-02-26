from collections import Counter

import pickle
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import Dataset, DataLoader

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_boolean


class SegregatedHomogeneousPatientDataset(Dataset):
    def __init__(self, graph_type, transform=None):
        super(SegregatedHomogeneousPatientDataset, self).__init__()
        if graph_type == 'large':
            filename = get_configurations_dtype_string(section='SETUP', key='LARGE_GRAPHS_FILENAME')
        else:
            filename = get_configurations_dtype_string(section='SETUP', key='SMALL_GRAPHS_FILENAME')
        self.data_list = pickle.load(open(filename, 'rb'))
        self.y = torch.as_tensor([graph_info[1] for graph_info in self.data_list])
        self.remove_knn_edges = get_configurations_dtype_boolean(section='TRAINING', key='REMOVE_KNN_EDGES', default_value=False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        graph, graph_label = self.data_list[item]
        return graph, graph_label

    @property
    def num_graphs(self):
        return len(self.data_list)


class SegregatedHeterogeneousPatientDataset(Dataset):
    def __init__(self, graph_type, transform=None):
        super(SegregatedHeterogeneousPatientDataset, self).__init__()
        if graph_type == 'large':
            filename = get_configurations_dtype_string(section='SETUP', key='LARGE_GRAPHS_FILENAME')
        else:
            filename = get_configurations_dtype_string(section='SETUP', key='SMALL_GRAPHS_FILENAME')
        self.data_list = pickle.load(open(filename, 'rb'))
        self.y = torch.as_tensor([graph_info[1] for graph_info in self.data_list])
        self.remove_knn_edges = get_configurations_dtype_boolean(section='TRAINING', key='REMOVE_KNN_EDGES', default_value=False)
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        heterogeneous_graph, graph_label = self.data_list[item]
        if self.remove_knn_edges:
            del heterogeneous_graph[('lesion', 'NN', 'lesion')]
        if self.transform is not None:
            heterogeneous_graph = self.transform(heterogeneous_graph)
        heterogeneous_graph = self.handle_isolated_edges(heterogeneous_graph)
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
        return len(self.data_list)


if __name__ == '__main__':
    transform = T.NormalizeFeatures()
    dataset = SegregatedHomogeneousPatientDataset(graph_type='small', transform=transform)
    print(dataset.num_graphs)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)
    for samp, label in loader:
        print(label.shape)
