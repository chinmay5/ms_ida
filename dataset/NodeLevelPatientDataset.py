from collections import Counter, defaultdict

import os
import pandas as pd
import pickle
import torch
import torch_geometric
from matplotlib import pyplot as plt
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import dropout_edge

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_int, \
    get_configurations_dtype_boolean
from utils.dataset_util import convert_to_pickle_file_name
from utils.training_utils import drop_nodes


class AbstractDataset(Dataset):
    def __init__(self):
        super(AbstractDataset, self).__init__()
        use_2y = get_configurations_dtype_boolean(section='SETUP', key='USE_2Y')
        self.node_drop = get_configurations_dtype_boolean(section='TRAINING', key='NODE_DROP')
        self.edge_drop = get_configurations_dtype_boolean(section='TRAINING', key='EDGE_DROP')
        self.random_node_translation = get_configurations_dtype_boolean(section='TRAINING', key='NODE_TRANS', default_value=False)
        if use_2y:
            self.annotated_data_pickle_location = convert_to_pickle_file_name(
                get_configurations_dtype_string(section='SETUP',
                                                key='ANNOTATED_DATA_2Y_CSV_LOCATION'))
            self.label_column = 'New_Lesions_2y_Label'
            self.graph_regr_column = 'New_Lesions_2y_volume_mm3'
            print("using 2y labels")
        else:
            self.annotated_data_pickle_location = convert_to_pickle_file_name(
                get_configurations_dtype_string(section='SETUP',
                                                key='ANNOTATED_DATA_1Y_CSV_LOCATION'))
            self.label_column = 'New_Lesions_1y_Label'
            self.graph_regr_column = f'New_Lesions_1y_volume_mm3'
        self.node_level_column = 'Volume'

    def compute_graph_category(self):
        diff_graph_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')
        graph_categ = []
        for idx in range(len(self.patient_list)):
            graph_name = f"{self.patient_list[idx]}.pt"
            heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))
            num_nodes = heterogeneous_graph['lesion'].x.shape[0]
            if num_nodes <= diff_graph_threshold:
                graph_categ.append(0)  # Small
            else:
                graph_categ.append(1)  # Large
        return torch.as_tensor(graph_categ)

    def apply_transform(self, orig_graph, use_edge_drop=True):
        if self.transform is not None:
            orig_graph = self.transform(orig_graph)

        augmented_graph = orig_graph.clone()
        if self.edge_drop:
            augmented_graph.edge_index, edge_mask = dropout_edge(orig_graph.edge_index, p=0.5,
                                                                 force_undirected=False,
                                                                 training=use_edge_drop)
            # Update the edge attributes by removing attributes for "dropped edges".
            if augmented_graph.edge_attr is not None and edge_mask is not None:
                augmented_graph.edge_attr = torch.masked_select(augmented_graph.edge_attr, edge_mask)
        if self.node_drop:
            augmented_graph = drop_nodes(augmented_graph)

        if self.random_node_translation:
            coord = augmented_graph.x[:, -3:]
            num_lesions = coord.shape[0]
            t = 0.02
            translated_coord = coord.new_empty((num_lesions, 3)).uniform_(-t, t)
            augmented_graph.x[:, -3:] += translated_coord
        return orig_graph, augmented_graph

    def __repr__(self):
        return f"Edge drop is {self.edge_drop}. Node drop is {self.node_drop}.\nLoading data from -> {self.annotated_data_pickle_location}"


class HomogeneousNodeLevelPatientDataset(AbstractDataset):
    def __init__(self, transform=None):
        super(HomogeneousNodeLevelPatientDataset, self).__init__()
        annotated_data = pd.read_pickle(self.annotated_data_pickle_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = annotated_data.loc[:, self.label_column]
        self.y_nodes = annotated_data.loc[:, self.node_level_column]
        self.y_regr = annotated_data.loc[:, self.graph_regr_column]
        self.graph_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_HETERO_DATASET_ROOT_FOLDER')
        self.remove_knn_edges = get_configurations_dtype_boolean(section='TRAINING', key='REMOVE_KNN_EDGES',
                                                                 default_value=False)
        self.transform = transform
        self.graph_catogory_label = self.compute_graph_category()

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        graph_name = f"{self.patient_list[item]}.pt"
        graph_label = int(self.y[item].item())
        node_labels = torch.as_tensor(self.y_nodes[item]).unsqueeze(1)
        graph_vol = self.y_regr[item].item()
        heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))
        if self.remove_knn_edges:
            del heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_attr
            del heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_index
            del heterogeneous_graph[('lesion', 'NN', 'lesion')]
        # Let us make it homogeneous
        homogeneous_graph = self.convert_to_homogeneous_graph(heterogeneous_graph)
        # Add the extra parameters required
        homogeneous_graph.node_labels = node_labels / 1e4
        homogeneous_graph.graph_vol = graph_vol / 1e4
        # Column normalize the features
        homogeneous_graph, homogeneous_graph_augmented = self.apply_transform(orig_graph=homogeneous_graph)
        return homogeneous_graph, homogeneous_graph_augmented, graph_label

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
        # Doing the same for the edge attribute
        if 'edge_attr' not in heterogeneous_graph:
            homogeneous_graph['edge_attr'] = torch.tensor([], dtype=torch.float)
        return homogeneous_graph

    @property
    def num_graphs(self):
        return len(self.y)


class HeterogeneousNodeLevelPatientDataset(AbstractDataset):
    def __init__(self, transform=None):
        super(HeterogeneousNodeLevelPatientDataset, self).__init__()
        annotated_data = pd.read_pickle(self.annotated_data_pickle_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = annotated_data.loc[:, self.label_column]
        self.y_nodes = annotated_data.loc[:, self.node_level_column]
        self.y_regr = annotated_data.loc[:, self.graph_regr_column]
        self.graph_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_HETERO_DATASET_ROOT_FOLDER')
        self.remove_knn_edges = get_configurations_dtype_boolean(section='TRAINING', key='REMOVE_KNN_EDGES',
                                                                 default_value=False)
        self.transform = transform
        self.graph_catogory_label = self.compute_graph_category()

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        graph_name = f"{self.patient_list[item]}.pt"
        graph_label = int(self.y[item].item())
        node_labels = torch.as_tensor(self.y_nodes[item]).unsqueeze(1)
        graph_vol = self.y_regr[item].item()
        heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))
        # Add the extra labels
        heterogeneous_graph.node_labels = node_labels / 1e4
        heterogeneous_graph.graph_vol = graph_vol / 1e4  # To normalize the range of values a bit
        heterogeneous_graph, heterogeneous_graph_augmented = self.apply_transform(orig_graph=heterogeneous_graph)
        heterogeneous_graph = self.handle_isolated_edges(heterogeneous_graph)
        # This step done later since it is much more convenient.
        # Remove the edges once all bases related to it are covered.
        if self.remove_knn_edges:
            del heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_attr
            del heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_index
            del heterogeneous_graph[('lesion', 'NN', 'lesion')]

        return heterogeneous_graph, heterogeneous_graph_augmented, graph_label

    def __repr__(self):
        return f"Heterogeneous graph with {self.y.size(0)} labels and {super.__repr__()}"

    def handle_isolated_edges(self, heterogeneous_graph):
        # We next handle if this was an isolated graph itself.
        # In such a scenario, we delete the edge_index
        # and allow it to become a heterogeneous graph.
        if 'edge_index' in heterogeneous_graph['lesion']:
            del heterogeneous_graph['lesion']['edge_index']
            heterogeneous_graph[('lesion', 'LesionLocation', 'lesion')].edge_index = torch.tensor([[], []],
                                                                                                  dtype=torch.long)
            heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_index = torch.tensor([[], []], dtype=torch.long)
            heterogeneous_graph[('lesion', 'Intercluster', 'lesion')].edge_index = torch.tensor([[], []], dtype=torch.long)
        # We also handle cases where one kind of edge is missing.
        if ('lesion', 'LesionLocation', 'lesion') not in heterogeneous_graph.edge_index_dict:
            heterogeneous_graph[('lesion', 'LesionLocation', 'lesion')].edge_index = torch.tensor([[], []],
                                                                                                  dtype=torch.long)
        if ('lesion', 'NN', 'lesion') not in heterogeneous_graph.edge_index_dict:
            heterogeneous_graph[('lesion', 'NN', 'lesion')].edge_index = torch.tensor([[], []], dtype=torch.long)
        if ('lesion', 'Intercluster', 'lesion') not in heterogeneous_graph.edge_index_dict:
            heterogeneous_graph[('lesion', 'Intercluster', 'lesion')].edge_index = torch.tensor([[], []], dtype=torch.long)
        return heterogeneous_graph

    @property
    def num_graphs(self):
        return len(self.y)


def separate_large_and_small_graphs():
    largeness_threshold = get_configurations_dtype_int(section='SETUP', key='LARGENESS_THRESHOLD')
    large_graphs_filename = get_configurations_dtype_string(section='SETUP', key='LARGE_GRAPHS_FILENAME')
    small_graphs_filename = get_configurations_dtype_string(section='SETUP', key='SMALL_GRAPHS_FILENAME')

    dataset = HomogeneousNodeLevelPatientDataset()
    small_patients_list, large_patients_list = [], []
    for idx in range(len(dataset)):
        candidate_graph = dataset[idx]
        if candidate_graph[0].x.shape[0] > largeness_threshold:
            large_patients_list.append(candidate_graph)
        else:
            small_patients_list.append(candidate_graph)
    pickle.dump(large_patients_list, open(large_graphs_filename, 'wb'))
    pickle.dump(small_patients_list, open(small_graphs_filename, 'wb'))
    print("large and small graph datasets successfully created ")


class KNNNodeLevelPatientDataset(HomogeneousNodeLevelPatientDataset):
    def __init__(self, transform=None):
        super(KNNNodeLevelPatientDataset, self).__init__()
        self.num_neighbours = get_configurations_dtype_int(section='SETUP', key='NUM_FEATURE_NEIGHBOURS')
        annotated_data = pd.read_pickle(self.annotated_data_pickle_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = annotated_data.loc[:, self.label_column]
        self.y_nodes = annotated_data.loc[:, self.node_level_column]
        self.y_regr = annotated_data.loc[:, self.graph_regr_column]
        self.graph_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_HETERO_DATASET_ROOT_FOLDER')

        self.transform = transform
        self.graph_catogory_label = self.compute_graph_category()

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        graph_name = f"{self.patient_list[item]}.pt"
        graph_label = int(self.y[item].item())
        node_labels = torch.as_tensor(self.y_nodes[item]).unsqueeze(1)
        graph_vol = self.y_regr[item].item()
        heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))

        # Let us make it homogeneous
        knn_homogeneous_graph = self.convert_to_homogeneous_graph(heterogeneous_graph)
        # Adding the extra labels
        knn_homogeneous_graph.node_labels = node_labels / 1e4
        knn_homogeneous_graph.graph_vol = graph_vol / 1e4  # To normalize the range of values a bit
        knn_homogeneous_graph, knn_homogeneous_graph_augmented = self.apply_transform(orig_graph=knn_homogeneous_graph)
        return knn_homogeneous_graph, knn_homogeneous_graph_augmented, torch.as_tensor([graph_label], dtype=torch.long)

    @property
    def num_graphs(self):
        return len(self.y)


class FullyConnectedNodeLevelDataset(HomogeneousNodeLevelPatientDataset):
    def __init__(self, transform=None):
        super(FullyConnectedNodeLevelDataset, self).__init__()
        annotated_data = pd.read_pickle(self.annotated_data_pickle_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = annotated_data.loc[:, self.label_column]
        self.y_nodes = annotated_data.loc[:, self.node_level_column]
        self.y_regr = annotated_data.loc[:, self.graph_regr_column]
        self.graph_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_HETERO_DATASET_ROOT_FOLDER')
        self.transform = transform
        self.graph_catogory_label = self.compute_graph_category()

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        graph_name = f"{self.patient_list[item]}.pt"
        graph_label = int(self.y[item].item())
        node_labels = torch.as_tensor(self.y_nodes[item]).unsqueeze(1)
        graph_vol = self.y_regr[item].item()
        heterogeneous_graph = torch.load(os.path.join(self.graph_folder, graph_name))
        # We will remove all the edges that are present in the graph
        del heterogeneous_graph[('lesion', 'NN', 'lesion')]
        del heterogeneous_graph[('lesion', 'LesionLocation', 'lesion')]
        # Let us make it homogeneous
        homogeneous_graph = self.convert_to_homogeneous_graph(heterogeneous_graph)
        # Generating all the edges for the fully connected graph
        N = homogeneous_graph.x.size(0)
        row = torch.arange(N).view(-1, 1).repeat(1, N).view(-1)
        col = torch.arange(N).view(-1, 1).repeat(N, 1).view(-1)
        edge_index = torch.stack([row, col], dim=0)
        homogeneous_graph.edge_index = edge_index
        homogeneous_graph.edge_attr = torch.ones(edge_index.shape[1], )
        # Adding the extra labels
        homogeneous_graph.node_labels = node_labels / 1e4
        homogeneous_graph.graph_vol = graph_vol / 1e4
        # Column normalize the features
        homogeneous_graph, homogeneous_graph_augmented = self.apply_transform(orig_graph=homogeneous_graph,
                                                                              use_edge_drop=True)
        # Let us add a dummy node at the end for each of the graphs.
        return homogeneous_graph, homogeneous_graph_augmented, torch.as_tensor([graph_label], dtype=torch.long)

    @property
    def num_graphs(self):
        return len(self.y)


def plot_histogram(num_nodes_to_label):
    fontsize = 15
    plt.rcParams.update({'font.size': fontsize})
    # Hacky fix since "small" comes later than "large" alphabetically
    num_nodes_to_label = {k: v for k, v in
                          sorted(num_nodes_to_label.items(), reverse=True)}

    plt.title("Num nodes vs. Occurrence")
    plt.bar(range(len(num_nodes_to_label)), [len(x) for x in num_nodes_to_label.values()],
            tick_label=list(num_nodes_to_label.keys()), color='b')
    plt.xlabel('Graph size')
    plt.ylabel("Occurrence")
    plt.show()


if __name__ == '__main__':
    # separate_large_and_small_graphs()
    # transform = T.NormalizeFeatures()
    dataset = FullyConnectedNodeLevelDataset()
    dataloader = torch_geometric.loader.DataLoader(dataset, batch_size=4, shuffle=False)
    edge_attr_info = []
    x_y_z_coord, x_y_z_coord_aug = [], []
    max_feat_val, min_feat_val = -float("inf"), float("inf")
    print(dataset)
    print(dataset[0][0].x.shape)
    print(dataset[0][1].x.shape)
    print(dataset.num_graphs)
    print(dataset.graph_catogory_label)
    all_labels = []
    graph_len_occurrence = defaultdict(int)
    num_nodes_to_label = defaultdict(list)
    small_large = defaultdict(list)
    thresh = 20
    min_vol, max_vol = float('inf'), 0
    min_vol_node, max_vol_node = float('inf'), 0
    for idx in range(len(dataset)):
        graph, aug_graph, label = dataset[idx]
        edge_attr_info.extend(graph.edge_attr.numpy().tolist())
        all_labels.append(label.item())
        graph_len_occurrence[graph.x.shape[0]] += 1
        if graph.edge_index.shape[1] <= thresh:
            small_large['small'].append(label.item())
        else:
            small_large['large'].append(label.item())
        # Same for the number of nodes
        if graph.x.shape[0] <= thresh // 2:
            num_nodes_to_label['small'].append(label.item())
        else:
            num_nodes_to_label['large'].append(label.item())
        # Also look at the volumes
        min_vol = min(min_vol, graph.graph_vol)
        max_vol = max(max_vol, graph.graph_vol)
        # Look at the lesion volumes as well
        min_vol_node = min(min_vol_node, graph.node_labels.squeeze().numpy().min())
        max_vol_node = max(max_vol_node, graph.node_labels.squeeze().numpy().max())
        # Verifying range of coordinate location values.
        x_y_z_coord.append(graph.x[:, -3:])
        max_feat_val = max(max_feat_val, graph.x.max())
        min_feat_val = min(min_feat_val, graph.x.min())
        # Similarly for the augmented graph.
        x_y_z_coord_aug.append(aug_graph.x[:, -3:])
    print(f"Minimum volume is {min_vol} while maximum is {max_vol}")
    print(f"Mininum node volume is {min_vol_node} while maximum is {max_vol_node}")
    print(Counter(all_labels))
    x_y_z_coord, x_y_z_coord_aug = torch.cat(x_y_z_coord), torch.cat(x_y_z_coord_aug)
    print("Min-max value in all coords are")
    print(f"Min x: {x_y_z_coord[:, 0].min()}, Min y: {x_y_z_coord[:, 1].min()} and Min z: {x_y_z_coord[:, 2].min()}")
    print(f"Max x: {x_y_z_coord[:, 0].max()}, Max y: {x_y_z_coord[:, 1].max()} and Max z: {x_y_z_coord[:, 2].max()}")
    print("Min-max value for augmented coords are")
    print(f"Min x: {x_y_z_coord_aug[:, 0].min()}, Min y: {x_y_z_coord_aug[:, 1].min()} and Min z: {x_y_z_coord_aug[:, 2].min()}")
    print(f"Max x: {x_y_z_coord_aug[:, 0].max()}, Max y: {x_y_z_coord_aug[:, 1].max()} and Max z: {x_y_z_coord_aug[:, 2].max()}")
    print("Feature value information")
    print(f"Maximum: {max_feat_val} and minimum {min_feat_val}")
    for graph_size, counts in graph_len_occurrence.items():
        print(f"{graph_size} ----- {counts}")
    plot_histogram(num_nodes_to_label)
    for size in ['small', 'large']:
        print(
            f"{size} has the edge distribution: {Counter(small_large[size])} and node distribution  {Counter(num_nodes_to_label[size])}")
    # Since we want to check the dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    data, data_aug, label = next(iter(dataloader))
    print(label)
    plt.hist(edge_attr_info, density=False, bins=10)
    plt.show()
