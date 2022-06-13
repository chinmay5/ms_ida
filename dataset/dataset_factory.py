from dataset.PatientDataset import HomogeneousPatientDataset, HeterogeneousPatientDataset
from dataset.SegregatedPatientDataset import SegregatedHeterogeneousPatientDataset, SegregatedHomogeneousPatientDataset
from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_string
import torch_geometric.transforms as T


def get_dataset():
    is_hetero = get_configurations_dtype_boolean(section='TRAINING', key='IS_HETERO')
    graph_type = get_configurations_dtype_string(section='SETUP', key='GRAPH_TYPE')
    use_segregated_dataset = False
    if graph_type == 'large' or graph_type == 'small':
        use_segregated_dataset = True
        print(f"Using segregated dataset of type {graph_type}")
    transform = None  # T.Compose([T.NormalizeFeatures()])
    if is_hetero:
        if use_segregated_dataset:
            return SegregatedHeterogeneousPatientDataset(graph_type=graph_type, transform=transform)
        return HeterogeneousPatientDataset(transform=transform)
    else:
        if use_segregated_dataset:
            return SegregatedHomogeneousPatientDataset(graph_type=graph_type, transform=transform)
        return HomogeneousPatientDataset(transform=transform)