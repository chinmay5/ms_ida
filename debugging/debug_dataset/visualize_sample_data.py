from dataset.PatientDataset import HeterogeneousPatientDataset, KNNPatientDataset
from environment_setup import get_configurations_dtype_int
from graph_creation.create_small_graph import visualize_heterogeneous_and_homogeneous_dataset, \
    visualize_homogeneous_dataset


def visualize_homogeneous_sample(sample, filename_prefix):
    # Taking the 5th sample from the dataset
    num_neighbours = get_configurations_dtype_int(section='SETUP', key='NUM_FEATURE_NEIGHBOURS')
    print(sample)
    visualize_homogeneous_dataset(filename=f"{filename_prefix + str(num_neighbours)}", homogeneous_dataset=sample,
                                  scan_to_patients=sample.graph_metadata.scan_to_patients)


def visualize_single_sample(sample, filename_prefix):
    # Taking the 5th sample from the dataset
    num_neighbours = get_configurations_dtype_int(section='SETUP', key='NUM_NEIGHBOURS')
    print(sample)
    visualize_heterogeneous_and_homogeneous_dataset(hetero_dataset=sample, scan_to_patients=sample.graph_metadata.scan_to_patients,
                                                    filename=f"{filename_prefix + str(num_neighbours)}")


if __name__ == '__main__':
    dataset = HeterogeneousPatientDataset()
    sample = dataset[4][0]
    filename_prefix = f'vizFifth_'
    visualize_single_sample(sample=sample, filename_prefix=filename_prefix)

    # Doing the same for KNN graph generated based on feature distances
    dataset = KNNPatientDataset()
    sample = dataset[4][0]
    filename_prefix = f'KNN_vizFifth_'
    visualize_homogeneous_sample(sample=sample, filename_prefix=filename_prefix)

