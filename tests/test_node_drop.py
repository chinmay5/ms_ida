import torch

from dataset.dataset_factory import get_dataset
from utils.training_utils import drop_nodes


def test_node_drop():
    dataset = get_dataset()
    sample_graph_data = dataset[0][0]
    updated_node = drop_nodes(sample_graph_data)
    print(updated_node)


if __name__ == '__main__':
    test_node_drop()
