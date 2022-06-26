import torch

from dataset.dataset_factory import get_dataset
from graph_models.model_factory import get_model
from model_training.eval_utils import eval_graph_len_acc, plot_results_based_on_graph_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test_cm_dict():
    dataset = get_dataset()
    sample_graph_data = dataset[0][0]
    model = get_model(model_type='gcn', hidden_dim=8,
                      num_layers=2, sample_graph_data=sample_graph_data)
    model.to(device)
    acc, size_cm_dict = eval_graph_len_acc(model, dataset)
    plot_results_based_on_graph_size(size_cm_dict, filename_acc='acc', filename_roc='roc')


if __name__ == '__main__':
    test_cm_dict()
