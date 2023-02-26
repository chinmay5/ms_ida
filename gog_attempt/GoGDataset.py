import torch
from torch.utils.data import Dataset
from torch_geometric.nn import knn_graph

from dataset.NodeLevelPatientDataset import KNNNodeLevelPatientDataset
from dataset.dataset_factory import get_dataset


class GoGDataset(Dataset):
    def __init__(self):
        super(GoGDataset, self).__init__()
        self.underlying_dataset = get_dataset()
        self.all_small_graphs = []
        self.load_data()

    @property
    def num_nodes(self):
        return len(self.underlying_dataset)

    def __len__(self):
        # This is a single graph
        return 1

    def __getitem__(self, item):
        return self.all_small_graphs

    def reset_dataset(self):
        self.load_data()

    def load_data(self):
        for idx in range(len(self.underlying_dataset)):
            data, _, label = self.underlying_dataset[idx]
            self.all_small_graphs.append((data, label))


if __name__ == '__main__':
    dataset_list = GoGDataset()
    print(dataset_list[0])
