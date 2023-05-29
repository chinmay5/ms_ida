import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GraphConv, SGConv, GINConv, EdgeConv, DynamicEdgeConv
from torch_geometric.nn.inits import uniform
from torch_scatter import segment_csr, scatter_max, scatter_add

from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_float, \
    get_configurations_dtype_string, device
from utils.deterministic_ops_utils import TransformerConvNoEdge
from utils.training_utils import count_parameters


def topk(x, ratio, batch, min_score=None, tol: float = 1e-7, use_ratio_as_numbers=False):
    if min_score is not None:
        # Make sure that we do not drop all nodes in a graph.
        scores_max = scatter_max(x, batch)[0].index_select(0, batch) - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero().view(-1)

    elif ratio is not None:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), int(num_nodes.max())

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,), -60000.0)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if ratio >= 1 and use_ratio_as_numbers:
            k = num_nodes.new_full((num_nodes.size(0),), int(ratio))
            k = torch.min(k, num_nodes)
        else:
            k = (float(ratio) * num_nodes.to(x.dtype)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    else:
        raise ValueError("At least one of 'min_score' and 'ratio' parameters "
                         "must be specified")

    return perm


class SimpleTanhAttn(nn.Module):
    def __init__(self, feat_dim, retention_ratio):
        super(SimpleTanhAttn, self).__init__()
        self.feat_dim = feat_dim
        self.retention_ratio = retention_ratio
        self.proj = nn.Parameter(torch.Tensor(1, feat_dim))
        print("Using sum pool for aggregation")

    def forward(self, logits, batch):
        out = {}
        score = torch.sigmoid((logits * self.proj).sum(dim=-1) / self.proj.norm(p=2, dim=-1))
        # Now, we will just use the top-k elements from the logit.
        # We will only use these top elements for the final prediction.
        top_elements = topk(score, ratio=self.retention_ratio, batch=batch, use_ratio_as_numbers=False)
        logits, batch = logits[top_elements], batch[top_elements]
        # The ptr tensor can be obtained as:-
        # `ptr` tensor begins with 0 and ends with total number of elements
        # The intermediate values indicate when there is a transition from elements of one batch to the other.
        transition_indices = (torch.where(batch[:-1] != batch[1:])[0] + 1)
        ptr_new = torch.empty(len(transition_indices) + 2, dtype=torch.long)
        ptr_new[0], ptr_new[-1] = 0, len(batch)
        ptr_new[1:-1] = transition_indices
        # We need to check if this should be `add` or `mean` pool.
        aggregated_features = deterministic_global_add_pool(logits, ptr_new.to(device))
        out['x'] = aggregated_features
        out['selected_nodes'] = top_elements
        return out

    def reset_parameters(self):
        # pyG uniform initialization operation.
        # Obtained from the top-k pool source code
        uniform(self.feat_dim, self.proj)


def deterministic_global_add_pool(x, batch=None):
    if batch is None:
        return x.sum(dim=-2, keepdim=x.dim() == 2)
    return segment_csr(x, batch, reduce='add')


def deterministic_global_mean_pool(x, batch=None):
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    return segment_csr(x, batch, reduce='mean')


class ParentHomogeneousGNN(nn.Module):
    """
    This is the parent class for all the Homogeneous convolution graphs.
    The parent class has definition of the `forward` method.
    All the children classes should define the self.convs() ModuleList.
    """

    def __init__(self, node_feature_dim, hidden_dim, retention_ratio, num_classes=2, edge_attr=True):
        super(ParentHomogeneousGNN, self).__init__()
        self.bns = None
        self.drs = None
        self.convs = None
        self.use_edge_attr = edge_attr
        self.node_feature_dim = node_feature_dim
        # self.node_location_map = nn.Embedding(4, node_feature_dim)
        # Let us use fixed positional encoding for our node feature map
        # self.node_location_map = PositionalEncoding(max_len=4, d_model=node_feature_dim)
        self.lin1 = nn.Linear(hidden_dim // 8, hidden_dim // 16)
        self.lin2 = nn.Linear(hidden_dim // 16, num_classes)
        self.use_sip = get_configurations_dtype_boolean(section='TRAINING',
                                                        key='USE_SIP')
        if self.use_sip:
            print("Creating SIP layer")
            self.global_pool_aggr = SimpleTanhAttn(hidden_dim // 8,
                                                   retention_ratio=retention_ratio)  # global_mean_pool  # SumAggregation()
        else:
            self.global_pool_aggr = deterministic_global_add_pool  # global_add_pool
        self.update_node_feat = nn.Linear(node_feature_dim, hidden_dim)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
        if isinstance(self.global_pool_aggr, SimpleTanhAttn):
            self.global_pool_aggr.reset_parameters()

    def forward(self, data):
        out = {}
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        if self.training and get_configurations_dtype_boolean(section='TRAINING', key='NODE_TRANS'):
            # Apply node augmentation
            coord, num_lesions = x[:, -3:], x.shape[0]
            t = get_configurations_dtype_float(section='TRAINING', key='MAX_TRANS')
            translated_coord = coord.new_empty((num_lesions, 3)).uniform_(-t, t)
            x[:, -3:] += translated_coord
        # Now we are also applying a mask on the node features.
        x = self.update_node_feat(x)

        # Take a look at the NodelevelPatientDataset file
        if edge_index is None:
            # sparse tensors
            mp_args = [data.adj_t.t()]
        elif self.use_edge_attr:
            mp_args = [edge_index, edge_attr]
        else:
            mp_args = [edge_index]
        prev = 0
        for idx, conv in enumerate(self.convs):
            x = x + prev  # skip conn
            if isinstance(conv, nn.Linear):
                x = F.leaky_relu(conv(x), negative_slope=0.2)
            elif isinstance(conv, DynamicEdgeConv):
                # We can not handle it for the time being :(
                torch.use_deterministic_algorithms(False)
                x = F.leaky_relu(conv(x), negative_slope=0.2)
                torch.use_deterministic_algorithms(True)
            else:
                x = F.leaky_relu(conv(x, *mp_args), negative_slope=0.2)
            if self.bns is not None:
                x = self.bns[idx](x)
            if self.drs is not None:
                x = self.drs[idx](x)
            prev = x

        self.x = x
        if self.use_sip:
            # pooling_out = self.global_pool_aggr(x, batch, data.ptr)
            pooling_out = self.global_pool_aggr(x, batch)
        elif self.global_pool_aggr == deterministic_global_add_pool:
            pooling_out = self.global_pool_aggr(x, data.ptr)
        else:
            pooling_out = self.global_pool_aggr(x, batch)

        if isinstance(pooling_out, dict):
            graph_level_feat = pooling_out.get('x')
            # out['weight_coeff'] = pooling_out.get('weights', None)
            out['selected_nodes'] = pooling_out['selected_nodes']
        else:
            graph_level_feat = pooling_out
        self.graph_level_feat = F.leaky_relu(self.lin1(graph_level_feat), negative_slope=0.2)
        # Need to normalize for loss to behave better.
        graph_level_feat = F.dropout(self.graph_level_feat, p=0.5, training=self.training)
        graph_label = self.lin2(graph_level_feat)
        out['graph_pred'] = graph_label
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}. Abstract parent class with only the forward method."

    def get_full_des(self):
        return super().__repr__()


class GCNHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(GCNHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes, retention_ratio=retention_ratio)
        convs = [GCNConv(in_channels=hidden_dim, out_channels=hidden_dim // 8)]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(GCNConv(in_channels=hidden_dim // 8, out_channels=hidden_dim // 8))
            bns.append(nn.LayerNorm(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GCN Convolutions."


class GATHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(GATHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes, edge_attr=False, retention_ratio=retention_ratio)
        convs = [GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim // 8)]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(GATv2Conv(in_channels=hidden_dim // 8, out_channels=hidden_dim // 8))
            bns.append(nn.LayerNorm(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GAT Convolutions."


class SAGEHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(SAGEHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes, retention_ratio=retention_ratio)
        convs = [SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim // 8)]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(SAGEConv(in_channels=hidden_dim // 8, out_channels=hidden_dim // 8))
            bns.append(nn.LayerNorm(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses SAGE Convolutions."


class EdgeHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(EdgeHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes, edge_attr=False, retention_ratio=retention_ratio)
        convs = [
            EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(in_features=2 * hidden_dim, out_features=hidden_dim // 8),
                    nn.ReLU(),
                    nn.Dropout()
                ),
                aggr='sum')
        ]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(EdgeConv(
                nn=nn.Sequential(
                    nn.Linear(2 * hidden_dim // 8, hidden_dim // 8),
                    nn.ReLU(),
                    nn.Dropout()
                ), aggr='sum', k=10)
            )
            bns.append(nn.LayerNorm(hidden_dim // 8))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Edge Convolutions."


class GINHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(GINHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes, edge_attr=False, retention_ratio=retention_ratio)
        convs = [GINConv(
            nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 8),
            ), train_eps=True,
        )]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim // 8, hidden_dim // 8),
                ), train_eps=True,
            ))
            bns.append(nn.LayerNorm(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GIN Convolutions."


class GraphConvHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(GraphConvHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                               num_classes=num_classes, retention_ratio=retention_ratio)
        convs = [GraphConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(GraphConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GraphConv Convolutions."


class SimpleConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(SimpleConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes, retention_ratio=retention_ratio)
        convs = [SGConv(in_channels=hidden_dim, out_channels=hidden_dim // 8, K=total_number_of_gnn_layers)]
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Simple Graph Convolutions."


class LinearModel(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2):
        super(LinearModel, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes, retention_ratio=retention_ratio)
        convs = [nn.Linear(hidden_dim, hidden_dim // 8)]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(nn.Linear(hidden_dim // 8, hidden_dim // 8))
            bns.append(nn.LayerNorm(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Fully Connected layers."


class TransformerLikeGATModel(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, retention_ratio, num_classes=2,
                 forward_expansion=2):
        super(TransformerLikeGATModel, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                                      num_classes=num_classes, edge_attr=False,
                                                      retention_ratio=retention_ratio)
        assert get_configurations_dtype_string(section='TRAINING', key='DATASET') == 'Fully', "Please use the " \
                                                                                              "Transformer like " \
                                                                                              "blocks only for the " \
                                                                                              "fully connected graph "

        heads = 1
        convs = [TransformerConvNoEdge(in_channels=hidden_dim, out_channels=hidden_dim // (8 * heads), heads=heads)]
        bns = [nn.LayerNorm(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(TransformerConvNoEdge(in_channels=hidden_dim // 8, out_channels=hidden_dim // (8 * heads),
                                               heads=heads))
            bns.append(nn.LayerNorm(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Transformer like Convolutions."


if __name__ == '__main__':
    # gcn_conv = TransformerLikeGATModel(hidden_dim=128, total_number_of_gnn_layers=2, node_feature_dim=768,
    #                                    num_classes=2)
    from torch_geometric.transforms import ToSparseTensor
    from torch_geometric import seed_everything
    from torch.backends import cudnn

    seed_everything(seed=42)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    from dataset.dataset_factory import get_dataset

    gcn_conv = SAGEHomConv(hidden_dim=128, total_number_of_gnn_layers=2, node_feature_dim=772,
                           num_classes=2, retention_ratio=1.0)
    gcn_conv.to(device)

    print(gcn_conv.get_full_des())
    # gcn_conv = LinearModel(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=512, num_classes=2)
    print(gcn_conv)
    gcn_conv.reset_parameters()

    dataset = get_dataset(transform=ToSparseTensor(attr='edge_attr'))
    # dataset = get_dataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    data, data_aug, label = next(iter(dataloader))
    print(data)
    # print(data_aug)
    output = gcn_conv(data.to(device))
    # output = gcn_conv(data)
    print(output['graph_pred'])
    print(output['graph_vol'].shape)
    # print(node_regr)
    # print(vol_regr)
    print(f"Total number of parameters = {count_parameters(gcn_conv) / 1e6}M")
    # print(MixupModel(gcn_conv)(data, data)[0].shape)
