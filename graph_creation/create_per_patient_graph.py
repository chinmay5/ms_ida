import copy
import os
import pandas as pd
import torch

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_int
from graph_creation.create_small_graph import visualize_heterogeneous_and_homogeneous_dataset
from graph_creation.graph_creation_utils import add_knn_nodes_to_df, make_heterogeneous_dataset


def split_graph_based_on_patients():
    csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
    df = pd.read_csv(csv_path, comment='#')
    patient_csv_root_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_CSV_ROOT_FOLDER')
    if os.path.exists(patient_csv_root_folder):
        print("Patient csv root folder exists. Reusing it!!!!!")
        return
    os.makedirs(patient_csv_root_folder, exist_ok=True)
    group_index_dict = df.groupby(by='Patient').indices
    for patient, csv_row_indices in group_index_dict.items():
        patient_df = copy.deepcopy(df.loc[csv_row_indices])
        # Finally, we would go ahead and save this patient info in a csv file
        patient_df.to_csv(os.path.join(patient_csv_root_folder, f"{patient}.csv"), index=False)
    print(f"Per patient csv files created by splitting {csv_path}")


def create_heterogeneous_graphs_for_one_patient(patient_df, patient_name, hetero_dataset_save_folder):
    num_neighbours = get_configurations_dtype_int(section='SETUP', key='NUM_NEIGHBOURS')
    patient_df_with_knn = add_knn_nodes_to_df(df=patient_df, k=num_neighbours)
    hetero_dataset = make_heterogeneous_dataset(all_scans_df_with_NN=patient_df_with_knn)
    # Finally, we would go ahead and save this dataset
    hetero_dataset_save_path = os.path.join(hetero_dataset_save_folder, f'{patient_name}.pt')
    torch.save(hetero_dataset, hetero_dataset_save_path)


def create_heterogeneous_graphs_for_all_patients():
    patient_csv_root_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_CSV_ROOT_FOLDER')
    patient_hetero_dataset_root_folder = get_configurations_dtype_string(section='SETUP',
                                                                         key='PATIENT_HETERO_DATASET_ROOT_FOLDER')
    os.makedirs(patient_hetero_dataset_root_folder, exist_ok=True)
    all_patient_csv_files = os.listdir(patient_csv_root_folder)
    for patient_csv_name in all_patient_csv_files:
        patient_df = pd.read_csv(os.path.join(patient_csv_root_folder, patient_csv_name))
        patient_name = patient_csv_name[:patient_csv_name.find(".csv")]
        create_heterogeneous_graphs_for_one_patient(patient_df=patient_df, patient_name=patient_name,
                                                    hetero_dataset_save_folder=patient_hetero_dataset_root_folder)


def sanitize_error_patients(patients_with_graph_formation_error):
    annotated_data_csv_folder = get_configurations_dtype_string(section='SETUP', key='ANNOTATED_DATA_CSV_LOCATION')
    annotated_scans = pd.read_csv(annotated_data_csv_folder)
    patients_with_annotations = set(annotated_scans.loc[:, 'Patient'].values.tolist())
    for patient in patients_with_graph_formation_error:
        if patient in patients_with_annotations:
            print(f"Patient {patient} does not have enough neighbours but has a label")


def test_graph_creation_for_one_patient(patient_name='m819631'):
    """
    We can use this method to test some of the edge cases.
    I could realize issues with the edge-cases using this method
    :param patient_name: Should correspond to the csv filename
    :return: None
    """
    patient_csv_root_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_CSV_ROOT_FOLDER')
    patient_df = pd.read_csv(os.path.join(patient_csv_root_folder, f"{patient_name}.csv"))
    temp_dataset_folder = 'temp_folder'
    os.makedirs(temp_dataset_folder, exist_ok=True)
    create_heterogeneous_graphs_for_one_patient(patient_df, patient_name, temp_dataset_folder)
    # Let us also plot this graph
    hetero_dataset_save_path = os.path.join(temp_dataset_folder, f'{patient_name}.pt')
    hetero_dataset = torch.load(hetero_dataset_save_path)
    visualize_heterogeneous_and_homogeneous_dataset(hetero_dataset, hetero_dataset.scan_to_patients,
                                                    filename=patient_name)


if __name__ == '__main__':
    split_graph_based_on_patients()
    create_heterogeneous_graphs_for_all_patients()
    # sanitize_error_patients(patients_with_graph_formation_error=patients_with_graph_formation_error)
    # test_graph_creation_for_one_patient()
