def convert_to_pickle_file_name(csv_filename):
    pickle_filename = csv_filename[:csv_filename.find(".csv")] + ".pkl"
    return pickle_filename


def get_patient_name_from_dataset(graph_dataset):
    return graph_dataset.graph_metadata.scan_to_patients[0][0]
