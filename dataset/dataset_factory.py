from dataset.NodeLevelPatientDataset import HomogeneousNodeLevelPatientDataset, KNNNodeLevelPatientDataset, \
    FullyConnectedNodeLevelDataset
from dataset.PatientDataset import HomogeneousPatientDataset, HeterogeneousPatientDataset, KNNPatientDataset, \
    FullyConnectedDataset
from dataset.SegregatedPatientDataset import SegregatedHeterogeneousPatientDataset, SegregatedHomogeneousPatientDataset
from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_string


def get_dataset(transform=None):
    dataset_type = get_configurations_dtype_string(section='TRAINING', key='DATASET')
    graph_type = get_configurations_dtype_string(section='SETUP', key='GRAPH_TYPE')
    is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION',
                                                             default_value=False)
    assert dataset_type in ['Homo', 'Hetero', 'KNN',
                            'Fully'], f"Acceptable options are Homo, Hetero, KNN & Fully, provided: {dataset_type}"
    use_segregated_dataset = False
    if graph_type == 'large' or graph_type == 'small':
        use_segregated_dataset = True
        print(f"Using segregated dataset of type {graph_type}")
    transform = transform  # T.Compose([T.NormalizeFeatures()])
    if dataset_type == 'Hetero':
        return create_heterogeneous_dataset(graph_type, transform, use_segregated_dataset)
    elif dataset_type == 'Homo':
        return create_homogeneous_dataset(graph_type, transform, use_segregated_dataset, is_node_level_dataset)
    elif dataset_type == 'KNN':
        print("No segregation supported for KNN yet")
        return create_knn_patient_dataset(transform, is_node_level_dataset)
    elif dataset_type == 'Fully':
        print("No segregation supported for Fully Connected dataset yet")
        return create_fully_connected_dataset(transform, is_node_level_dataset)
    else:
        raise AttributeError("Invalid dataset option selected")


def create_fully_connected_dataset(transform, is_node_level_dataset):
    if is_node_level_dataset:
        print("Using node level fully connected dataset")
        return FullyConnectedNodeLevelDataset(transform=transform)
    return FullyConnectedDataset(transform=transform)


def create_knn_patient_dataset(transform, is_node_level_dataset):
    if is_node_level_dataset:
        print("Using KNN node level dataset")
        return KNNNodeLevelPatientDataset(transform=transform)
    return KNNPatientDataset(transform=transform)


def create_homogeneous_dataset(graph_type, transform, use_segregated_dataset, is_node_level_dataset):
    print("Creating homogeneous dataset")
    if use_segregated_dataset:
        if is_node_level_dataset:
            raise NotImplementedError("Segregation for Node level dataset is not available")
        return SegregatedHomogeneousPatientDataset(graph_type=graph_type, transform=transform)
    else:
        if is_node_level_dataset:
            print("Using the node level dataset")
            return HomogeneousNodeLevelPatientDataset(transform=transform)
        return HomogeneousPatientDataset(transform=transform)


def create_heterogeneous_dataset(graph_type, transform, use_segregated_dataset):
    if use_segregated_dataset:
        return SegregatedHeterogeneousPatientDataset(graph_type=graph_type, transform=transform)
    return HeterogeneousPatientDataset(transform=transform)
