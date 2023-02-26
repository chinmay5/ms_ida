import os
import pandas as pd

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_boolean


def get_min_max_lesions_size(column_to_process):
    csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
    df = pd.read_csv(csv_path)
    return df[[column_to_process]].min().item(), df[[column_to_process]].max().item()


def generate_labels_for_patients_and_individual_lesions(label_column, annotated_data_csv_name, unlabeled_data_csv_name=None):
    # node level label column
    node_level_label_column = 'Volume'
    vol_regr_column = f'{label_column}_volume_mm3'
    patient_csv_root_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_CSV_ROOT_FOLDER')
    all_patient_csv_files = os.listdir(patient_csv_root_folder)
    # min_individual_tumor_val, max_individual_tumor_val = get_min_max_lesions_size(
    #     column_to_process=node_level_label_column)
    # min_new_tumor_val, max_new_tumor_val = get_min_max_lesions_size(column_to_process=vol_regr_column)
    # We would iterate through each of them and then add the results in a dataframe.
    patient_label_volume_tuple_list = []
    patient_unlabel_volume_tuple_list = []
    unlabeled_count = 0
    # The original dataframe is sorted by the GlobalLesionId.
    # We can use the same in order to ensure consistency between the lesion number and its corresponding volume.
    for patient_csv_name in all_patient_csv_files:
        patient_csv = pd.read_csv(os.path.join(patient_csv_root_folder, patient_csv_name))
        label = patient_csv.loc[:, label_column].values.tolist()
        # Check if all the labels are identical
        assert len(set(label)) == 1, "Scan has multiple labels associated. Please check"

        # Now get the associated lesion volume
        # volume_list = ((patient_csv.loc[:, node_level_label_column].values - min_individual_tumor_val) / (
        #             max_individual_tumor_val - min_individual_tumor_val)).tolist()
        volume_list = patient_csv.loc[:, node_level_label_column].values.tolist()
        # new_tumor_volume_list = ((patient_csv.loc[:, vol_regr_column].values - min_new_tumor_val) / (
        #             max_new_tumor_val - min_new_tumor_val)).tolist()
        new_tumor_volume_list = patient_csv.loc[:, vol_regr_column].values.tolist()
        assert len(set(new_tumor_volume_list)) == 1, "Scan has multiple different tumor volumes. Please check"
        new_tumor_vol = new_tumor_volume_list[0]

        patient_name = patient_csv_name[:patient_csv_name.find(".csv")]
        patient_label = label[0]
        # We would skip the scan if it does not have label as 0 or 1.
        # if patient_label not in [0, 1]:
        # Needs to be changed since Bene changed the dataset
        if patient_label == -1:
            patient_unlabel_volume_tuple_list.append((patient_name, patient_label, volume_list, new_tumor_vol))
            unlabeled_count += 1
            continue
        # We need to map the labels to 0: Absent and 1: Present
        if patient_label != 0:
            print(f"Changing {patient_label} to 1")
            patient_label = 1
        patient_label_volume_tuple_list.append((patient_name, patient_label, volume_list, new_tumor_vol))
    print(f"Got a total of {unlabeled_count} unlabeled scans. Using it for semi-supervised learning.")
    # Finally, we can go ahead and store the values. This would act as our annotated dataset file
    data = pd.DataFrame.from_records(data=patient_label_volume_tuple_list,
                                     columns=['Patient', f'{label_column}_Label', 'Volume', f'{vol_regr_column}'])
    data.to_pickle(annotated_data_csv_name)
    if unlabeled_data_csv_name is not None:
        unlabeled_data = pd.DataFrame.from_records(data=patient_unlabel_volume_tuple_list,
                                         columns=['Patient', f'{label_column}_Label', 'Volume', f'{vol_regr_column}'])
        unlabeled_data.to_pickle(unlabeled_data_csv_name)
    return data


def generate_labels_for_patients(label_column, annotated_data_csv_name):
    patient_csv_root_folder = get_configurations_dtype_string(section='SETUP', key='PATIENT_CSV_ROOT_FOLDER')
    all_patient_csv_files = os.listdir(patient_csv_root_folder)
    # We would iterate through each of them and then add the results in a dataframe.
    patient_label_tuple_list = []
    for patient_csv_name in all_patient_csv_files:
        patient_csv = pd.read_csv(os.path.join(patient_csv_root_folder, patient_csv_name))
        label = patient_csv.loc[:, label_column].values.tolist()
        # Check if all the labels are identical
        assert len(set(label)) == 1, "Scan has multiple labels associated. Please check"
        patient_name = patient_csv_name[:patient_csv_name.find(".csv")]
        patient_label = label[0]
        # We would skip the scan if it does not have label as 0 or 1.
        if patient_label not in [0, 1]:
            continue
        patient_label_tuple_list.append((patient_name, patient_label))
    # Finally, we can go ahead and store the values. This would act as our annotated dataset file
    data = pd.DataFrame.from_records(data=patient_label_tuple_list, columns=['Patient', f'{label_column}_Label'])
    data.to_csv(annotated_data_csv_name, index=False)


if __name__ == '__main__':
    use_2y = get_configurations_dtype_boolean(section='SETUP', key='USE_2Y')
    if use_2y:
        label_column = 'New_Lesions_2y'
        annotated_data_csv_name = get_configurations_dtype_string(section='SETUP', key='ANNOTATED_DATA_2Y_CSV_LOCATION')
        unlabeled_data_csv_name = get_configurations_dtype_string(section='SETUP', key='UNLABELED_DATA_2Y_CSV_LOCATION')
    else:
        label_column = 'New_Lesions_1y'
        annotated_data_csv_name = get_configurations_dtype_string(section='SETUP',
                                                                  key='ANNOTATED_DATA_1Y_CSV_LOCATION')
        unlabeled_data_csv_name = get_configurations_dtype_string(section='SETUP',
                                                                  key='UNLABELED_DATA_1Y_CSV_LOCATION')
    generate_labels_for_patients(label_column=label_column, annotated_data_csv_name=annotated_data_csv_name)
    print("Creating a pickle file to conveniently store the lists")
    annotated_data_pickle_name = annotated_data_csv_name[:annotated_data_csv_name.find(".csv")] + ".pkl"
    unlabeled_data_pickle_name = None
    if unlabeled_data_csv_name is not None:
        unlabeled_data_pickle_name = unlabeled_data_csv_name[:annotated_data_csv_name.find(".csv")] + ".pkl"
    print(f"Generating {annotated_data_pickle_name}")

    generate_labels_for_patients_and_individual_lesions(label_column=label_column,
                                                        annotated_data_csv_name=annotated_data_pickle_name,
                                                        unlabeled_data_csv_name=unlabeled_data_pickle_name)
