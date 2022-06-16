from dataset.PatientDataset import HeterogeneousPatientDataset
from graph_creation.create_small_graph import visualize_heterogeneous_dataset


def visualize_single_sample():
    dataset = HeterogeneousPatientDataset()
    # Taking the 5th sample from the dataset
    sample = dataset[4][0]
    visualize_heterogeneous_dataset(hetero_dataset=sample, scan_to_patients=sample.graph_metadata.scan_to_patients,
                                    filename='vizFifth')


if __name__ == '__main__':
    visualize_single_sample()
