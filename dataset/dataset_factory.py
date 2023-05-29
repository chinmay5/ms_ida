from dataset.PatientDataset import HomogeneousNodeLevelPatientDataset, KNNNodeLevelPatientDataset, \
    FullyConnectedNodeLevelDataset
from environment_setup import get_configurations_dtype_string


def get_dataset(transform=None):
    dataset_type = get_configurations_dtype_string(section='TRAINING', key='DATASET')

    assert dataset_type in ['Homo', 'Hetero', 'KNN',
                            'Fully'], f"Acceptable options are Homo, Hetero, KNN & Fully, provided: {dataset_type}"
    transform = transform
    if dataset_type == 'Homo':
        return create_homogeneous_dataset(transform)
    elif dataset_type == 'KNN':
        print("No segregation supported for KNN yet")
        return create_knn_patient_dataset(transform)
    elif dataset_type == 'Fully':
        print("No segregation supported for Fully Connected dataset yet")
        return create_fully_connected_dataset(transform)
    else:
        raise AttributeError("Invalid dataset option selected")


def create_fully_connected_dataset(transform):
    print("Using node level fully connected dataset")
    return FullyConnectedNodeLevelDataset(transform=transform)


def create_knn_patient_dataset(transform):
    return KNNNodeLevelPatientDataset(transform=transform)


def create_homogeneous_dataset(transform):
    print("Creating homogeneous dataset")
    return HomogeneousNodeLevelPatientDataset(transform=transform)
