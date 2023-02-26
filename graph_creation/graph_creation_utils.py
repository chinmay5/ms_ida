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
from utils.training_utils import normalize_features

print(
    "WARNING!!!!!\n A change had to be made to ignore the last scan.\n This happened because the new csv does not "
    "contain the last scan.\nPlease remove this code block as soon as the csv issue is resolved.")
ssl_radiomic_features = np.load(get_configurations_dtype_string(section='SETUP', key='SSL_FEATURES_NUMPY_PATH'))
# radiomic_features = np.load(get_configurations_dtype_string(section='SETUP', key='RADIOMIC_FEATURES_NUMPY_PATH'))
# TODO: Remve this ASAP
ssl_radiomic_features = ssl_radiomic_features[:-1, :]
# ssl_radiomic_features, radiomic_features = ssl_radiomic_features[:-1, :], radiomic_features[:-1, :]
# concatenated_features = np.concatenate((radiomic_features, ssl_radiomic_features), axis=1)
#
# normalize_features(concatenated_features)


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
    edge_index, edge_type, edge_attr, edge_type_names_dict = [], [], [], dict()
    # Make this scan names instead since that is what we eventually want for plotting
    all_scans_df_with_NN.sort_values(by=['GlobalLesionID'], inplace=True)
    scan_to_patients = {idx: item for idx, item in enumerate(all_scans_df_with_NN.values)}
    df_to_scan_index = {item[1]: idx for idx, item in enumerate(all_scans_df_with_NN.values)}
    compute_edges(all_possible_permutations, all_scans_df_with_NN, df_to_scan_index, edge_index, edge_type,
                  edge_attr, edge_type_names_dict)
    hetero_dataset = get_heterogeneous_dataset(edge_index, edge_type, edge_attr, edge_type_names_dict, scan_to_patients)
    # Let us add scan_to_patients info to this data object
    # We use the metadata class for it to not throw errors during batching
    scan_to_patient_metadata = GraphMetadata()
    scan_to_patient_metadata.scan_to_patients = scan_to_patients
    hetero_dataset.graph_metadata = scan_to_patient_metadata
    return hetero_dataset


def get_heterogeneous_dataset(edge_index, edge_type, edge_attr, edge_type_names_dict, scan_to_patients):
    dataset = to_pyg_dataset(edge_index=edge_index, edge_attr=edge_attr, scan_to_patients=scan_to_patients)
    # We convert the edge type to a tensor
    edge_type = torch.LongTensor(edge_type)
    # sort the names so that they retain the order
    edge_type_names_dict = dict(sorted(edge_type_names_dict.items(), key=lambda x: x[1]))
    hetero_data = dataset.to_heterogeneous(node_type=torch.zeros(dataset.x.shape[0], dtype=torch.long),
                                           edge_type=edge_type, edge_type_names=list(edge_type_names_dict.keys()),
                                           node_type_names=['lesion'])
    return hetero_data


def to_pyg_dataset(edge_index, edge_attr, scan_to_patients):
    edge_index = torch.LongTensor(edge_index).t()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    node_clusters = None
    # We can use the node features obtained from SSL pretraining: credit Bene
    global_lesion_indices_for_small_graph_nodes = []
    for _, metadata_values in scan_to_patients.items():
        global_lesion_indices_for_small_graph_nodes.append(metadata_values[1])
    # We have the indices and now we can get the node features
    x = torch.as_tensor(ssl_radiomic_features, dtype=torch.float)
    # Using concatenation of radiomic and SSL features
    # x = torch.as_tensor(concatenated_features, dtype=torch.float)
    add_if_enhancing_tumor_as_feature = get_configurations_dtype_boolean(section='SETUP',
                                                                         key='ADD_ENHANCING_TUMOR_LOCATION')
    csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
    all_patients_df = pd.read_csv(csv_path)
    raw_coord_location = torch.as_tensor(np.asarray(all_patients_df[['x', 'y', 'z']]), dtype=torch.float) / 128
    if add_if_enhancing_tumor_as_feature:
        enhancing_tumor_as_numpy_feature = np.asarray(all_patients_df[['Enhancing']])
        enhancing_tumor_as_pytorch_feature = torch.as_tensor(enhancing_tumor_as_numpy_feature, dtype=torch.float)
        x = torch.cat([x, enhancing_tumor_as_pytorch_feature], dim=1)
    if_add_tumor_location_as_feature = get_configurations_dtype_boolean(section='SETUP',
                                                                        key='ADD_NODE_LOCATION_AS_FEATURE')
    if if_add_tumor_location_as_feature:
        csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
        all_patients_df = pd.read_csv(csv_path)
        lesion_location_as_numpy_feature = np.asarray(all_patients_df[['LesionLocation']])
        lesion_location_as_pytorch_feature = torch.as_tensor(lesion_location_as_numpy_feature, dtype=torch.float)
        x = torch.cat([x, lesion_location_as_pytorch_feature], dim=1)
        # Let us also add cluster centres as an extra term in the Data object.
        # This way, we can try our recursive coarsening approach.
        node_clusters = torch.as_tensor(lesion_location_as_numpy_feature, dtype=torch.long)[
            global_lesion_indices_for_small_graph_nodes].squeeze(-1)
    # We add the locations at the end, just to make our lives easier.
    x = torch.cat([x, raw_coord_location], dim=1)
    # We would select the indices based on the number of lesions for a specific patient
    x = x[global_lesion_indices_for_small_graph_nodes]
    data = Data(x=x, edge_index=edge_index.contiguous(), edge_attr=edge_attr)
    data.cluster = node_clusters
    return data


def compute_inverse_weighted_l1_distance(scan1, scan2):
    coordinate_wise_distance_diff = np.asarray(scan1[['x', 'y', 'z']].tolist()) - np.array(
        scan2[['x', 'y', 'z']].tolist())
    eucledian_distance = np.linalg.norm(coordinate_wise_distance_diff,
                                        ord=2) / 1e3  # Normalize values. Input is 32^3. 1e4 is a decent normalizing factor.
    return np.exp(-(eucledian_distance / 0.01))


def include_lesion_location_edges(all_possible_permutations, all_scans_df, df_to_scan_index, edge_index, edge_type,
                                  edge_attr, edge_type_names_dict):
    for df_index_tuple in tqdm(all_possible_permutations):
        scan1, scan2 = all_scans_df.iloc[df_index_tuple[0]], all_scans_df.iloc[df_index_tuple[1]]
        if selection_criterion(sample1=scan1, sample2=scan2):
            # Add index locations based on our mapping
            edge_index.append(get_scan_idx_from_scan(scan1=scan1, scan2=scan2, df_to_scan_index=df_to_scan_index))
            edge_type.append(0)
            # Include the l1 distance between edges as an edge-attribute
            edge_attr.append(compute_inverse_weighted_l1_distance(scan1, scan2))
            # To keep track of the different edges we are generating
            edge_type_names_dict[('lesion', 'LesionLocation', 'lesion')] = 0
        # Check if the two can be added via the inter-cluster edges
        elif inter_cluster_selection_criterion(sample1=scan1, sample2=scan2):
            # Add index locations based on our mapping
            edge_index.append(get_scan_idx_from_scan(scan1=scan1, scan2=scan2, df_to_scan_index=df_to_scan_index))
            edge_type.append(1)
            # Include the l1 distance between edges as an edge-attribute
            edge_attr.append(compute_inverse_weighted_l1_distance(scan1, scan2))
            # To keep track of the different edges we are generating
            edge_type_names_dict[('lesion', 'Intercluster', 'lesion')] = 1
        # NOTE: pyG conversion from homo-hetero would do an "enumerate check"
        # on edge type to compare it with the edge_type_names.
        # As such, if the values are not contiguous, the edges would not be correctly assigned.
        # We return the next usable edge_type identifier
        return 2


def include_knn_edges(edge_index, edge_type, edge_attr, edge_type_names_dict, scan_with_NN, edge_type_idx):
    for index, row in scan_with_NN.iterrows():
        # We can check for the number of NN per patient
        num_neighbours = len([x for x in scan_with_NN.columns if 'NN_' in x])
        for neighbour_idx in range(num_neighbours):
            edge_index.append((index, row[f'NN_{neighbour_idx}']))
            edge_type.append(edge_type_idx)
            edge_attr.append(
                compute_inverse_weighted_l1_distance(scan1=row, scan2=scan_with_NN.iloc[row[f'NN_{neighbour_idx}']]))
            edge_type_names_dict[('lesion', 'NN', 'lesion')] = edge_type_idx


def compute_edges(all_possible_permutations, all_scans_df_with_NN, df_to_scan_index, edge_index, edge_type,
                  edge_attr, edge_type_names_dict):
    edge_type_idx = 0
    if get_configurations_dtype_boolean(section='SETUP', key='CREATE_LESION_EDGES'):
        print("Creating lesion edges")
        edge_type_idx = include_lesion_location_edges(all_possible_permutations, all_scans_df_with_NN, df_to_scan_index, edge_index,
                                      edge_type, edge_attr,
                                      edge_type_names_dict)
    if get_configurations_dtype_boolean(section='SETUP', key='CREATE_KNN_EDGES'):
        print("Creating KNN edges")
        # Now, we would select the second kind of edges based on the Nearest Neighbours.
        include_knn_edges(edge_index, edge_type, edge_attr, edge_type_names_dict, all_scans_df_with_NN, edge_type_idx)


def get_scan_idx_from_scan(scan1, scan2, df_to_scan_index):
    return df_to_scan_index[scan1[1]], df_to_scan_index[scan2[1]]


def selection_criterion(sample1, sample2):
    return sample1['LesionLocation'] == sample2['LesionLocation']


def inter_cluster_selection_criterion(sample1, sample2):
    return sample1['LesionLocation'] != sample2['LesionLocation'] and \
           compute_inverse_weighted_l1_distance(scan1=sample1, scan2=sample2) >= 0.4
