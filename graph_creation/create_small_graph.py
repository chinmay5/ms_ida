import os

import pandas as pd

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_int
from graph_creation.graph_creation_utils import add_knn_nodes_to_df, make_heterogeneous_dataset
from utils.viz_utils import plot_3d_graph, plot_heterogeneous_3d_graph, to_networkx_fail_safe


def create_subset_for_patient_782():
    csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
    df = pd.read_csv(csv_path)
    subset_782_csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV_PATIENT_782')
    subset_782_parent_folder = subset_782_csv_path[:subset_782_csv_path.rfind("/")]
    os.makedirs(subset_782_parent_folder, exist_ok=True)
    df.iloc[:46, ].to_csv(subset_782_csv_path, index=False)
    print("Small subset for patient 782 created")


def add_nearest_neighbours_and_save_df(df, k=10):
    # Let us create a copy of this csv file first
    df = add_knn_nodes_to_df(df, k)
    # Finally, we can store this csv file.
    # The second round of indices are generated based on the Nearest Neighbours only.
    subset_782_csv_with_nn_path = get_configurations_dtype_string(section='SETUP',
                                                                  key='RAW_METADATA_CSV_PATIENT_782_WITH_NN_CSV')
    df.to_csv(subset_782_csv_with_nn_path, index=False)
    return df


def create_heterogeneous_dataset_and_visualize(all_scans_df):
    # To compute permutations, we would need to use the `index` of the dataframe.
    hetero_dataset = make_heterogeneous_dataset(all_scans_df)
    print(hetero_dataset)
    visualize_heterogeneous_and_homogeneous_dataset(hetero_dataset, hetero_dataset.graph_metadata.scan_to_patients)


def visualize_heterogeneous_and_homogeneous_dataset(hetero_dataset, scan_to_patients, filename='small_viz'):
    plot_heterogeneous_3d_graph(hetero_dataset=hetero_dataset, scan_to_patients=scan_to_patients,
                                out_file=f'{filename}_hetero_graph.html')
    homogeneous_dataset = hetero_dataset.to_homogeneous()
    visualize_homogeneous_dataset(filename, homogeneous_dataset, scan_to_patients)


def visualize_homogeneous_dataset(filename, homogeneous_dataset, scan_to_patients):
    nx_graph = to_networkx_fail_safe(data=homogeneous_dataset)
    plot_3d_graph(edge_list=nx_graph.edges(), m_graph=nx_graph, scan_to_patients=scan_to_patients,
                  out_file=f'{filename}_graph.html')


if __name__ == '__main__':
    create_subset_for_patient_782()
    subset_782_csv = pd.read_csv(get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV_PATIENT_782'))
    # We need to run this code block only when we want to update the original CSV with more neighbours.
    num_neighbours = get_configurations_dtype_int(section='SETUP', key='NUM_NEIGHBOURS')
    df_with_nearest_neighbour = add_nearest_neighbours_and_save_df(df=subset_782_csv, k=num_neighbours)
    create_heterogeneous_dataset_and_visualize(all_scans_df=df_with_nearest_neighbour)
