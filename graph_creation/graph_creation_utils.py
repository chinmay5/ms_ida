import copy
import itertools
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from tqdm import tqdm

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_boolean

ssl_radiomic_features = np.load(get_configurations_dtype_string(section='SETUP', key='SSL_FEATURES_NUMPY_PATH'))


class GraphMetadata(object):
    def __init__(self):
        self.scan_to_patient = None


def fix_scan_order_global():
    subset_782_csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
    scans = pd.read_csv(subset_782_csv_path)
    # Ensuring that the values are always sorted
    scans.sort_values(by=['GlobalLesionID'], inplace=True)
    # Scan ordering is important since we want to decide which scan should be interpreted as scan0 in our graph
    scan_to_patients = {idx: item for idx, item in enumerate(scans.values)}
    df_to_scan_index = {item[1]: idx for idx, item in enumerate(scans.values)}
    pickle.dump(scan_to_patients, open('scan_to_patients.pkl', 'wb'))
    pickle.dump(df_to_scan_index, open('df_to_scan_index.pkl', 'wb'))


def add_knn_nodes_to_df(df, k):
    df = copy.deepcopy(df)
    features = df.loc[:, ['x', 'y', 'z']]
    try:
        nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(features)
        nearest_indices = nbrs.kneighbors(features)[1]
        # Let us create the extra columns to store NN
        for idx in range(k):
            df.loc[:, f"NN_{idx}"] = nearest_indices[:, idx + 1]
    except ValueError as e:
        # We do not have enough neighbours for the this patient. So,
        # We just return all the lesions and connect everything to everything
        all_candidates = df.index.values
        max_neighbours_available = max(all_candidates)
        neighbours_without_self = np.array([all_candidates[all_candidates != x] for x in df.index])
        # Let us create the extra columns to store NN
        for idx in range(max_neighbours_available):
            # Since the self loop is already removed, we need not explicitly ignore it unlike the other case.
            df.loc[:, f"NN_{idx}"] = neighbours_without_self[:, idx]
    return df


def make_heterogeneous_dataset(all_scans_df_with_NN):
    all_possible_permutations = itertools.permutations(all_scans_df_with_NN.index, r=2)
    edge_index, edge_type, edge_type_names_dict = [], [], dict()
    # Make this scan names instead since that is what we eventually want for plotting
    all_scans_df_with_NN.sort_values(by=['GlobalLesionID'], inplace=True)
    scan_to_patients = {idx: item for idx, item in enumerate(all_scans_df_with_NN.values)}
    df_to_scan_index = {item[1]: idx for idx, item in enumerate(all_scans_df_with_NN.values)}
    compute_edges(all_possible_permutations, all_scans_df_with_NN, df_to_scan_index, edge_index, edge_type,
                  edge_type_names_dict)
    hetero_dataset = get_heterogeneous_dataset(edge_index, edge_type, edge_type_names_dict, scan_to_patients)
    # Let us add scan_to_patients info to this data object
    # We use the metadata class for it to not throw errors during batching
    scan_to_patient_metadata = GraphMetadata()
    scan_to_patient_metadata.scan_to_patients = scan_to_patients
    hetero_dataset.graph_metadata = scan_to_patient_metadata
    return hetero_dataset


def get_heterogeneous_dataset(edge_index, edge_type, edge_type_names_dict, scan_to_patients):
    dataset = to_pyg_dataset(edge_index=edge_index, scan_to_patients=scan_to_patients)
    # We convert the edge type to a tensor
    edge_type = torch.LongTensor(edge_type)
    # sort the names so that they retain the order
    edge_type_names_dict = dict(sorted(edge_type_names_dict.items(), key=lambda x: x[1]))
    hetero_data = dataset.to_heterogeneous(node_type=torch.zeros(dataset.x.shape[0], dtype=torch.long),
                                           edge_type=edge_type, edge_type_names=list(edge_type_names_dict.keys()),
                                           node_type_names=['lesion'])
    return hetero_data


def to_pyg_dataset(edge_index, scan_to_patients):
    edge_index = torch.LongTensor(edge_index).t()
    # We can use the node features obtained from SSL pretraining: credit Bene
    global_lesion_indices_for_small_graph_nodes = []
    for _, metadata_values in scan_to_patients.items():
        global_lesion_indices_for_small_graph_nodes.append(metadata_values[1])
    # We have the indices and now we can get the node features
    x = torch.as_tensor(ssl_radiomic_features, dtype=torch.float)
    add_if_enhancing_tumor_as_feature = get_configurations_dtype_boolean(section='SETUP', key='ADD_ENHANCING_TUMOR_LOCATION')
    if add_if_enhancing_tumor_as_feature:
        csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
        all_patients_df = pd.read_csv(csv_path)
        enhancing_tumor_as_numpy_feature = np.asarray(all_patients_df[['Enhancing']])
        enhancing_tumor_as_pytorch_feature = torch.as_tensor(enhancing_tumor_as_numpy_feature, dtype=torch.float)
        x = torch.cat([x, enhancing_tumor_as_pytorch_feature], dim=1)
    # We would select the indices based on the number of lesions for a specific patient
    x = x[global_lesion_indices_for_small_graph_nodes]
    data = Data(x=x, edge_index=edge_index.contiguous())
    return data


def include_lesion_location_edges(all_possible_permutations, all_scans_df, df_to_scan_index, edge_index, edge_type,
                                  edge_type_names_dict):
    for df_index_tuple in tqdm(all_possible_permutations):
        scan1, scan2 = all_scans_df.iloc[df_index_tuple[0]], all_scans_df.iloc[df_index_tuple[1]]
        if selection_criterion(sample1=scan1, sample2=scan2):
            # Add index locations based on our mapping
            edge_index.append(get_scan_idx_from_scan(scan1=scan1, scan2=scan2, df_to_scan_index=df_to_scan_index))
            edge_type.append(0)
            # To keep track of the different edges we are generating
            edge_type_names_dict[('lesion', 'LesionLocation', 'lesion')] = 0


def include_knn_edges(edge_index, edge_type, edge_type_names_dict, scan_with_NN):
    for index, row in scan_with_NN.iterrows():
        # We can check for the number of NN per patient
        num_neighbours = len([x for x in scan_with_NN.columns if 'NN_' in x])
        for neighbour_idx in range(num_neighbours):
            edge_index.append((index, row[f'NN_{neighbour_idx}']))
            edge_type.append(1)
            edge_type_names_dict[('lesion', 'NN', 'lesion')] = 1


def compute_edges(all_possible_permutations, all_scans_df_with_NN, df_to_scan_index, edge_index, edge_type,
                  edge_type_names_dict):
    include_lesion_location_edges(all_possible_permutations, all_scans_df_with_NN, df_to_scan_index, edge_index,
                                  edge_type,
                                  edge_type_names_dict)
    # Now, we would select the second kind of edges based on the Nearest Neighbours.
    include_knn_edges(edge_index, edge_type, edge_type_names_dict, all_scans_df_with_NN)


def get_scan_idx_from_scan(scan1, scan2, df_to_scan_index):
    return df_to_scan_index[scan1[1]], df_to_scan_index[scan2[1]]


def selection_criterion(sample1, sample2):
    return sample1['LesionLocation'] == sample2['LesionLocation']
