from dataset.PatientDataset import HeterogeneousPatientDataset
from environment_setup import get_configurations_dtype_int
from graph_creation.create_small_graph import visualize_heterogeneous_dataset


def visualize_single_sample():
    dataset = HeterogeneousPatientDataset()
    # Taking the 5th sample from the dataset
    num_neighbours = get_configurations_dtype_int(section='SETUP', key='NUM_NEIGHBOURS')
    sample = dataset[4][0]
    print(sample)
    visualize_heterogeneous_dataset(hetero_dataset=sample, scan_to_patients=sample.graph_metadata.scan_to_patients,
                                    filename=f'vizFifth_{num_neighbours}')


if __name__ == '__main__':
    visualize_single_sample()
