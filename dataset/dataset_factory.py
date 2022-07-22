from dataset.PatientDataset import HomogeneousPatientDataset, HeterogeneousPatientDataset, KNNPatientDataset
from dataset.SegregatedPatientDataset import SegregatedHeterogeneousPatientDataset, SegregatedHomogeneousPatientDataset
from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_string
import torch_geometric.transforms as T


def get_dataset():
    dataset_type = get_configurations_dtype_string(section='TRAINING', key='DATASET')
    graph_type = get_configurations_dtype_string(section='SETUP', key='GRAPH_TYPE')
    assert dataset_type in ['Homo', 'Hetero', 'KNN'], "Acceptable options are Homo, Hetero, KNN"
    use_segregated_dataset = False
    if graph_type == 'large' or graph_type == 'small':
        use_segregated_dataset = True
        print(f"Using segregated dataset of type {graph_type}")
    transform = None  # T.Compose([T.NormalizeFeatures()])
    if dataset_type == 'Hetero':
        if use_segregated_dataset:
            return SegregatedHeterogeneousPatientDataset(graph_type=graph_type, transform=transform)
        return HeterogeneousPatientDataset(transform=transform)
    elif dataset_type == 'Homo':
        if use_segregated_dataset:
            return SegregatedHomogeneousPatientDataset(graph_type=graph_type, transform=transform)
        return HomogeneousPatientDataset(transform=transform)
    elif dataset_type == 'KNN':
        print("No segregation supported for KNN yet")
        return KNNPatientDataset(transform=transform)
    else:
        raise AttributeError("Invalid dataset option selected")