import os
import pickle

import torch
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import k_hop_subgraph

from environment_setup import get_configurations_dtype_string
from utils.viz_utils import to_networkx_fail_safe, plot_3d_graph


def predict_on_graph(graph, is_node_level_dataset, model):
    with torch.no_grad():
        out = model(graph)
        return out


def predict_with_grad_on_graph(graph, is_node_level_dataset, model):
    # The same prediction logic. However, in this case, we are going to allow gradients to flow.
    # The gradient flow becomes important for GNNExplainer classes
    out = model(graph)
    return out


def get_fold_from_index(graph_idx):
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    num_folds = pickle.load(open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'rb'))
    print(f"Using a pre-defined {num_folds} fold split. Done for easy reproducibility.")
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'rb'))
    for fold, indices in enumerate(test_indices):
        if graph_idx in indices:
            return fold


class DummyObject:
    def __init__(self):
        pass


class CaptumCompatibleModelWrapper(torch.nn.Module):
    def __init__(self, nn_model):
        super(CaptumCompatibleModelWrapper, self).__init__()
        print(
            f"IMPORTANT: This is a hack. Please make sure the forward function of present here\n aligns with that of {nn_model}")
        print("REMEMBER: This is your responsibility and the framework is not going to handle it on its own.")
        self.nn_model = nn_model
        self.nn_model.eval()

    def forward(self, x, edge_index):
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        dataobject = DummyObject()
        dataobject.x = x
        dataobject.edge_index = edge_index
        dataobject.batch = batch
        original_output, _, _ = self.nn_model(dataobject)

        # Now we perform our modified pass
        node_features = x[:, :-1]
        node_pos_info = self.nn_model.node_location_map(x[:, -1].to(torch.long) - 1)
        # x = F.normalize((node_features + node_pos_info), dim=1)
        # Since a similar pattern is used while computing pos_encoding in Transformers
        x = F.dropout(node_features + node_pos_info, p=0.1, training=False)
        if self.nn_model.is_node_level_dataset:
            x = self.nn_model.update_node_feat(x)
            x1 = self.nn_model.bn(x)
            x1 = F.relu(x1)
            x1 = self.nn_model.dr(x1)
            # The output values are in the range (0, 1)
            vol_regr = torch.sigmoid(self.nn_model.regress_lesion_volume(x1))

        for idx, conv in enumerate(self.nn_model.convs):
            x = F.leaky_relu(conv(x, edge_index), negative_slope=0.2)
            if self.nn_model.bns is not None:
                x = self.nn_model.bns[idx](x)
            if self.nn_model.drs is not None:
                x = self.nn_model.drs[idx](x)

        graph_level_feat = global_add_pool(x, batch)
        graph_level_feat = F.leaky_relu(self.nn_model.lin1(graph_level_feat), negative_slope=0.2)
        graph_level_feat = F.dropout(graph_level_feat, p=0.5, training=False)
        graph_label = self.nn_model.lin2(graph_level_feat)
        if self.nn_model.is_node_level_dataset:
            assert torch.allclose(original_output, graph_label), "Something wrong in the computation"
        return graph_label


# These two functions are specifically designed for monkey-patching.
class ExplainabilityViz(object):
    def __init__(self, graph_data):
        super(ExplainabilityViz, self).__init__()
        self.graph_data = graph_data

    def visualize_subgraph(self, node_idx, edge_index, edge_mask, y=None,
                           threshold=None, edge_y=None, node_alpha=None,
                           seed=10, viz_name=None, is_captum=False, **kwargs):
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

        # We first plot the original graph
        self.plot_graph(out_file=f'original_{viz_name}.html')
        self.graph_data.edge_index = edge_index
        prefix = 'Captum' if is_captum else 'GNN_explainer'
        self.plot_graph(out_file=f'{prefix}_{viz_name}.html')

    def plot_graph(self, out_file):
        nx_graph = to_networkx_fail_safe(data=self.graph_data)
        plot_3d_graph(edge_list=nx_graph.edges(), m_graph=nx_graph,
                      scan_to_patients=self.graph_data.graph_metadata.scan_to_patients,
                      out_file=out_file)
