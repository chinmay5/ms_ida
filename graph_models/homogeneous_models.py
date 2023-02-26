import math

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATv2Conv, SAGEConv, GraphConv, SGConv, global_add_pool, global_mean_pool, \
    TransformerConv, GINConv, EdgeConv, DynamicEdgeConv
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.topk_pool import topk
from torch_geometric.utils import dropout_edge, to_dense_batch, softmax
from torch_scatter import segment_csr

from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_float, \
    get_configurations_dtype_string, device
from utils.deterministic_ops_utils import TransformerConvNoEdge, STESigmoidThresholding
from utils.model_utils import PositionalEncoding
from utils.training_utils import count_parameters


class GumbelSoftmax(nn.Module):

    def __init__(self, feat_dim, eps=0.05):
        super(GumbelSoftmax, self).__init__()
        self.eps = eps
        self.sigmoid = nn.Sigmoid()
        self.linear_proj = nn.Linear(feat_dim, 1)

    def gumbel_sample(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = torch.log(uniform_samples_tensor + eps) - torch.log(
            1 - uniform_samples_tensor + eps)
        return gumble_samples_tensor

    def gumbel_softmax(self, logits):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        gsamples = self.gumbel_sample(logits.clone().detach())
        logits = logits + gsamples
        soft_samples = self.sigmoid(logits / self.eps)
        return soft_samples, logits

    def forward(self, logits, batch_indicator):
        # We need to make sure that the graphs do not end up interferring
        # with each other while computing attention masks
        logits = self.linear_proj(logits).squeeze(-1)
        if not self.training:
            out_hard = (logits >= 0).float()
            return out_hard, torch.tensor([0]), batch_indicator
        out_soft, prob_soft = self.gumbel_softmax(logits)
        out_hard = ((out_soft >= 0.5).float() - out_soft).detach() + out_soft
        return out_hard, out_soft, batch_indicator

    def reset_parameters(self):
        self.linear_proj.reset_parameters()


class SimpleTanhAttn(nn.Module):
    def __init__(self, feat_dim):
        super(SimpleTanhAttn, self).__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Parameter(torch.Tensor(1, feat_dim))

    def forward(self, logits, batch):
        score = torch.sigmoid((logits * self.proj).sum(dim=-1) / self.proj.norm(p=2, dim=-1))
        # Now, we will just use the top-k elements from the logit.
        # We will only use these top elements for the final prediction.
        top_elements = topk(score, ratio=0.2, batch=batch)
        logits, batch = logits[top_elements], batch[top_elements]
        # The ptr tensor can be obtained as:-
        # `ptr` tensor begins with 0 and ends with total number of elements
        # The intermediate values indicate when there is a transition from elements of one batch to the other.
        transition_indices = (torch.where(batch[:-1] != batch[1:])[0] + 1)
        ptr_new = torch.empty(len(transition_indices) + 2, dtype=torch.long)
        ptr_new[0], ptr_new[-1] = 0, len(batch)
        ptr_new[1:-1] = transition_indices
        return deterministic_global_add_pool(logits, ptr_new.to(device))

    def reset_parameters(self):
        # pyG uniform initialization operation.
        # Obtained from the top-k pool source code
        uniform(self.feat_dim, self.proj)


class SSLGCNEncoder(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim):
        super(SSLGCNEncoder, self).__init__()
        # self.conv1 = GINConv(
        #         nn.Sequential(
        #             nn.Linear(node_feature_dim + 1, 2 * hidden_dim),
        #         ), train_eps=True
        # )
        self.conv1 = GCNConv(node_feature_dim, 2 * hidden_dim)
        self.norm = nn.LayerNorm(2 * hidden_dim)
        # self.conv2 = GINConv(
        #     nn.Sequential(
        #         nn.Linear(2 * hidden_dim, hidden_dim),
        #     ), train_eps=True
        # )
        self.conv2 = GCNConv(2 * hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index).relu()
        x = self.norm(x)
        return self.conv2(x, data.edge_index)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.norm.reset_parameters()
        self.conv2.reset_parameters()

    def __repr__(self):
        return f"{self.__class__.__name__}_{self.hidden_dim}."

    def get_full_des(self):
        return super().__repr__()


def deterministic_global_add_pool(x, batch=None):
    if batch is None:
        return x.sum(dim=-2, keepdim=x.dim() == 2)
    return segment_csr(x, batch, reduce='add')


def deterministic_global_mean_pool(x, batch=None):
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    return segment_csr(x, batch, reduce='mean')


def deterministic_global_mean_pool_non_zero(x, batch=None):
    # Remove the nodes that are 0
    row, col = torch.nonzero(x, as_tuple=True)
    non_zero_rows = torch.unique(row)
    x = x[non_zero_rows, :]
    if batch is None:
        return x.mean(dim=-2, keepdim=x.dim() == 2)
    batch = batch[non_zero_rows]
    return segment_csr(x, batch, reduce='mean')


class SimpleAggr(nn.Module):
    def __init__(self, feat_dim, is_hard_masking):
        super(SimpleAggr, self).__init__()
        self.remove_nodes = is_hard_masking
        self.diff_thresholder = STESigmoidThresholding()
        # We can also explore making this portion a bit "stronger" as well.
        # self.coeff = nn.Sequential(
        #     nn.Linear(feat_dim, feat_dim * 2),
        #     nn.ReLU(),
        #     nn.Linear(feat_dim * 2, 1)
        # )
        self.coeff = nn.Sequential(
            nn.Linear(feat_dim, 1)
        )
        self.use_sip_attn = get_configurations_dtype_boolean(section='TRAINING', key='USE_SIP_ATTN')
        print(f"Using SIP Attention: {self.use_sip_attn}")
        if self.use_sip_attn:
            self.self_attn = SimpleTanhAttn(feat_dim=feat_dim)  # AttentionHead(input_dim=feat_dim)

    def forward(self, x, batch, ptr=None):
        out = {}
        # weights = torch.softmax(self.coeff(x), dim=0)
        # weights = torch.sigmoid(self.coeff(x))
        # Does adding the relu help us?
        if self.use_sip_attn:
            weights = self.self_attn(logits=x, batch=batch)
            non_discrete_weights = weights.clone()
            # Needed for the multiplication operation
            weights.unsqueeze_(-1)
        else:
            weights = torch.sigmoid(self.coeff(x))
            non_discrete_weights = weights.clone()
        # Discretize the weights
        selection_mask = weights  # self.diff_thresholder(weights).to(x.device)
        # if not self.training:
        #     updated_weights = []
        #     index = [(beg.item(), end.item()) for beg, end in zip(ptr[:-1], ptr[1:])]
        #     x_per_item, weight_per_item_new, weight_per_item_orig = [x[beg:end] for beg, end in index], [
        #         selection_mask[beg:end] for beg, end in index], [weights[beg:end] for beg, end in index]
        #     for x1, weight1, weight_orig1 in zip(x_per_item, weight_per_item_new, weight_per_item_orig):
        #         # If all the nodes are "off", pick 1/3 of the total.
        #         if weight1.sum() == 0:
        #             _, indices = torch.topk(weight_orig1, max(1, x1.shape[0] // 3), dim=0)
        #             weights_new = torch.zeros_like(weight1)
        #             weights_new[indices] = 1  #weight_orig1[indices]
        #             updated_weights.append(weights_new)
        #         else:
        #             updated_weights.append(weight1)
        #     selection_mask = torch.cat(updated_weights)
        x = torch.multiply(x, selection_mask)
        x = deterministic_global_add_pool(x, ptr)
        # x = deterministic_global_mean_pool(x, ptr)
        out['x'] = x
        out['weights'] = non_discrete_weights
        out['on_ratio'] = (non_discrete_weights >= 0.5).sum() / non_discrete_weights.shape[0]
        out['remove_nodes'] = self.remove_nodes
        return out

    def reset_parameters(self):
        for layer in self.coeff.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        self.coeff[-1].bias.data.fill_(0.5)
        if self.use_sip_attn:
            self.self_attn.reset_parameters()


class ParentHomogeneousGNN(nn.Module):
    """
    This is the parent class for all the Homogeneous convolution graphs.
    The parent class has definition of the `forward` method.
    All the children classes should define the self.convs() ModuleList.
    """

    def __init__(self, node_feature_dim, hidden_dim, num_classes=2, edge_attr=True):
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
            is_hard_masking = get_configurations_dtype_boolean(section='TRAINING', key='HARD_MASK')
            # self.global_pool_aggr = SimpleAggr(hidden_dim // 8, is_hard_masking)  # global_mean_pool  # SumAggregation()
            self.global_pool_aggr = SimpleTanhAttn(hidden_dim // 8)  # global_mean_pool  # SumAggregation()
        else:
            self.global_pool_aggr = deterministic_global_mean_pool  # global_add_pool
        self.is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP',
                                                                      key='PERFORM_NODE_LEVEL_PREDICTION')
        self.process_only_small = get_configurations_dtype_boolean(section='TRAINING',
                                                                   key='SKIP_LARGER', default_value=False)
        if self.is_node_level_dataset:
            self.update_node_feat = nn.Linear(node_feature_dim, hidden_dim)
            # self.bn = nn.BatchNorm1d(hidden_dim)
            self.dr = nn.Dropout(p=0.5)
            self.regress_lesion_volume = nn.Linear(hidden_dim, 1)
            self.lin_new_lesion_regr = nn.Linear(hidden_dim // 16, 1)

    def reset_parameters(self):
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()
        # for conv in self.convs:
        #     conv.reset_parameters()
        # if self.bns is not None:
        #     for bn in self.bns:
        #         bn.reset_parameters()
        # self.lin1.reset_parameters()
        # self.lin2.reset_parameters()
        # if isinstance(self.global_pool_aggr, SimpleAggr):
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
        if self.is_node_level_dataset:
            x = self.update_node_feat(x)
            # x1 = self.bn(x)
            x1 = F.relu(x)
            x1 = self.dr(x1)
            # The output values are in the range (0, 1)
            # vol_regr = torch.sigmoid(self.regress_lesion_volume(x1))
            vol_regr = torch.log(1 + torch.exp(self.regress_lesion_volume(x1)))
            # These updated features would be the ones our GNN uses for predictions.
            # x = x.detach()
        # edge_index, edge_mask = dropout_edge(edge_index, p=0.3,
        #                              force_undirected=False,
        #                              training=self.training)
        # edge_attr = edge_attr[edge_mask]
        if self.process_only_small and x.shape[0] >= 10:
            # Discard the graph structure
            edge_index = torch.tensor([[], []], dtype=torch.long).to(edge_index.device)
            edge_attr = torch.tensor([], dtype=torch.float).to(edge_index.device)
        # Remember to also drop attributes of the edges.
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
        # Now, the resnet connection
        # x = x + self.downsample_block(x_slip)
        # graph_level_feat = global_add_pool(x, batch)
        # graph_level_feat = self.lin1(graph_level_feat)
        # TODO: Check this part
        self.x = x
        # graph_level_feat = global_add_pool(x, batch)
        # index = 4 * data.batch + (data.cluster.squeeze() - 1)
        # graph_level_feat = self.global_pool_aggr(x, index=index)
        # Adjusting for cases where less than 4 clusters are present.
        # if graph_level_feat.shape[0] != batch_size * 4:
        # This happens only when the last graph does not have enough clusters.
        # In other words, the max of index is not (batch_size * 4 - 1).
        # We can solve it by padding the output with 0s
        # graph_level_feat = torch.nn.functional.pad(graph_level_feat,
        #                                            (0, 0, 0, (batch_size * 4) - int(index.max() + 1)), "constant",
        #                                            0)
        # graph_level_feat = graph_level_feat.view(batch_size, -1)
        # We pick most different 25 % samples.
        # idx = fps(x, batch, ratio=0.25, random_start=False)
        # x, batch = x[idx], batch[idx]
        if self.use_sip:
            # pooling_out = self.global_pool_aggr(x, batch, data.ptr)
            pooling_out = self.global_pool_aggr(x, batch)
        elif self.global_pool_aggr == deterministic_global_mean_pool:
            pooling_out = self.global_pool_aggr(x, data.ptr)
        else:
            pooling_out = self.global_pool_aggr(x, batch)

        if isinstance(pooling_out, dict):
            graph_level_feat = pooling_out['x']
            out['weight_coeff'] = pooling_out['weights']
            out['remove_nodes'] = pooling_out['remove_nodes']
            out['on_ratio'] = pooling_out['on_ratio']
        else:
            graph_level_feat = pooling_out
        self.graph_level_feat = F.leaky_relu(self.lin1(graph_level_feat), negative_slope=0.2)
        # Need to normalize for loss to behave better.
        graph_level_feat = F.dropout(self.graph_level_feat, p=0.5, training=self.training)
        graph_label = self.lin2(graph_level_feat)
        out['graph_pred'] = graph_label
        if self.is_node_level_dataset:
            # Using the log-sum-exp trick to ensure numerical stability
            pred_out = self.lin_new_lesion_regr(graph_level_feat).squeeze()
            graph_regr = torch.log(1 + torch.exp(pred_out - pred_out.max())) + pred_out.max()
            out['node_vol'] = vol_regr
            out['graph_vol'] = graph_regr
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}. Abstract parent class with only the forward method."

    def get_full_des(self):
        return super().__repr__()


class GCNHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GCNHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes)
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
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GATHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes, edge_attr=False)
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
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(SAGEHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes)
        convs = [SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim // 8)]
        bns = [nn.BatchNorm1d(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(SAGEConv(in_channels=hidden_dim // 8, out_channels=hidden_dim // 8))
            bns.append(nn.BatchNorm1d(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses SAGE Convolutions."


class EdgeHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(EdgeHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes, edge_attr=False)
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
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GINHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes, edge_attr=False)
        convs = [GINConv(
            nn.Sequential(
                nn.Linear(in_features=hidden_dim, out_features=hidden_dim // 8),
            ), train_eps=True,
        )]
        bns = [nn.BatchNorm1d(hidden_dim // 8)]
        drs = [nn.Dropout(p=0.5)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(GINConv(
                nn.Sequential(
                    nn.Linear(hidden_dim // 8, hidden_dim // 8),
                ), train_eps=True,
            ))
            bns.append(nn.BatchNorm1d(hidden_dim // 8))
            drs.append(nn.Dropout(p=0.5))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.drs = nn.ModuleList(drs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GIN Convolutions."


class GraphConvHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GraphConvHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                               num_classes=num_classes)
        convs = [GraphConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(GraphConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GraphConv Convolutions."


class SimpleConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(SimpleConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes)
        convs = [SGConv(in_channels=hidden_dim, out_channels=hidden_dim // 8, K=total_number_of_gnn_layers)]
        # bns = [nn.BatchNorm1d(hidden_dim // 8)]
        self.convs = nn.ModuleList(convs)
        # self.bns = nn.ModuleList(bns)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Simple Graph Convolutions."


class LinearModel(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(LinearModel, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes)
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
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2, forward_expansion=2):
        super(TransformerLikeGATModel, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                                      num_classes=num_classes, edge_attr=False)
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
                           num_classes=2)
    gcn_conv.to(device)
    # gcn_conv = SSLGCNEncoder(node_feature_dim=769, hidden_dim=32)

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
