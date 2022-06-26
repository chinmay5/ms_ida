import os

import pickle
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch_geometric.nn import GNNExplainer
from torch_geometric.utils import k_hop_subgraph
from tqdm import tqdm

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string
from graph_models.model_factory import get_model
from utils.viz_utils import to_networkx_fail_safe, plot_3d_graph


class CustomGNNExplainer(GNNExplainer):
    def __init__(self, model, epochs: int = 100, lr: float = 0.01,
                 num_hops=None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, graph_data=None, **kwargs):
        super(CustomGNNExplainer, self).__init__(model=model, epochs=epochs, lr=lr,
                                                 num_hops=num_hops, return_type=return_type,
                                                 feat_mask_type=feat_mask_type, allow_edge_mask=allow_edge_mask,
                                                 log=log, kwargs=kwargs)
        self.graph_data = graph_data

    def explain_graph(self, x, edge_index, **kwargs):

        self.model.eval()
        self.__clear_masks__()

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        self.graph_data.batch = batch

        # Get the initial prediction.
        with torch.no_grad():
            out = self.model(self.graph_data)
            if self.return_type == 'regression':
                prediction = out
            else:
                log_logits = self.__to_log_prob__(out)
                pred_label = log_logits.argmax(dim=-1)

        self.__set_masks__(x, edge_index)
        self.to(x.device)
        if self.allow_edge_mask:
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            # The teeny-tiny change we need to make
            out = self.model(self.graph_data)
            if self.return_type == 'regression':
                loss = self.__loss__(-1, out, prediction)
            else:
                log_logits = self.__to_log_prob__(out)
                loss = self.__loss__(-1, log_logits, pred_label)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        edge_mask = self.edge_mask.detach().sigmoid()

        self.__clear_masks__()
        return node_feat_mask, edge_mask

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, edge_y=None, node_alpha=None,
                           seed=10, **kwargs):
        r"""Visualizes the subgraph given an edge mask
        :attr:`edge_mask`.

        Args:
            node_idx (int): The node id to explain.
                Set to :obj:`-1` to explain graph.
            edge_index (LongTensor): The edge indices.
            edge_mask (Tensor): The edge mask.
            y (Tensor, optional): The ground-truth node-prediction labels used
                as node colorings. All nodes will have the same color
                if :attr:`node_idx` is :obj:`-1`.(default: :obj:`None`).
            threshold (float, optional): Sets a threshold for visualizing
                important edges. If set to :obj:`None`, will visualize all
                edges with transparancy indicating the importance of edges.
                (default: :obj:`None`)
            edge_y (Tensor, optional): The edge labels used as edge colorings.
            node_alpha (Tensor, optional): Tensor of floats (0 - 1) indicating
                transparency of each node.
            seed (int, optional): Random seed of the :obj:`networkx` node
                placement algorithm. (default: :obj:`10`)
            **kwargs (optional): Additional arguments passed to
                :func:`nx.draw`.

        :rtype: :class:`matplotlib.axes.Axes`, :class:`networkx.DiGraph`
        """
        import matplotlib.pyplot as plt

        assert edge_mask.size(0) == edge_index.size(1)

        if node_idx == -1:
            hard_edge_mask = torch.BoolTensor([True] * edge_index.size(1),
                                              device=edge_mask.device)
            subset = torch.arange(edge_index.max().item() + 1,
                                  device=edge_index.device)
            y = None

        else:
            # Only operate on a k-hop subgraph around `node_idx`.
            subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
                node_idx, self.num_hops, edge_index, relabel_nodes=True,
                num_nodes=None, flow=self.__flow__())

        edge_mask = edge_mask[hard_edge_mask]

        if threshold is not None:
            edge_mask = (edge_mask >= threshold)
            selected_edges = edge_index.T[edge_mask]
            edge_index = selected_edges.T

        if y is None:
            y = torch.zeros(edge_index.max().item() + 1,
                            device=edge_index.device)
        else:
            y = y[subset].to(torch.float) / y.max().item()

        # We first plot the original graph
        self.plot_graph(out_file='original_graph.html')
        self.graph_data.edge_index = edge_index
        self.plot_graph(out_file='GNN_explainer_graph.html')

    def plot_graph(self, out_file):
        nx_graph = to_networkx_fail_safe(data=self.graph_data)
        plot_3d_graph(edge_list=nx_graph.edges(), m_graph=nx_graph,
                      scan_to_patients=self.graph_data.graph_metadata.scan_to_patients,
                      out_file=out_file)


def get_test_split_indices():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
    return test_indices


def execute_gnn_explainer(model, optimizer, graph, graph_label):
    # Training a dummy model
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        logits = model(graph)
        loss = loss_fn(logits, torch.as_tensor(graph_label, dtype=torch.long).unsqueeze(0))
        loss.backward()
        optimizer.step()

    explainer = CustomGNNExplainer(model, epochs=200, return_type='raw', graph_data=graph)
    node_feat_mask, edge_mask = explainer.explain_graph(graph.x, graph.edge_index)
    # node_idx -1 is used to explain the entire graph
    explainer.visualize_subgraph(node_idx=-1, edge_index=graph.edge_index, edge_mask=edge_mask, threshold=0.2)
    plt.show()


def visualize_model():
    best_config_dict = {
        "hidden": 128,
        "num_layers": 2
    }
    dataset = get_dataset()
    sample_graph_data = dataset[0][0]
    graph_label = dataset[0][1]
    model = get_model(model_type='gcn', hidden_dim=best_config_dict["hidden"],
                      num_layers=best_config_dict["num_layers"], sample_graph_data=sample_graph_data)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    batch = torch.zeros(sample_graph_data.x.shape[0], dtype=int, device=sample_graph_data.x.device)
    sample_graph_data.batch = batch
    execute_gnn_explainer(model=model, optimizer=optimizer, graph=sample_graph_data, graph_label=graph_label)


if __name__ == '__main__':
    visualize_model()
