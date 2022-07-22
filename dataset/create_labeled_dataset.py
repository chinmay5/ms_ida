import os
import pandas as pd

from environment_setup import get_configurations_dtype_string, get_configurations_dtype_boolean


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
    return data


if __name__ == '__main__':
    use_2y = get_configurations_dtype_boolean(section='SETUP', key='USE_2Y')
    if use_2y:
        label_column = 'New_Lesions_2y'
        annotated_data_csv_name = get_configurations_dtype_string(section='SETUP', key='ANNOTATED_DATA_2Y_CSV_LOCATION')
    else:
        label_column = 'New_Lesions_1y'
        annotated_data_csv_name = get_configurations_dtype_string(section='SETUP',
                                                                    key='ANNOTATED_DATA_2Y_CSV_LOCATION')
    generate_labels_for_patients(label_column=label_column, annotated_data_csv_name=annotated_data_csv_name)
