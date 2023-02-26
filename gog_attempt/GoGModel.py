import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import DynamicEdgeConv, SAGEConv

from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_float, device
from gog_attempt.GoGDataset import GoGDataset
from graph_models.homogeneous_models import deterministic_global_add_pool


class ParentHomogeneousGraphLevelGNN(nn.Module):
    """
    This is the parent class for all the Homogeneous convolution graphs.
    The parent class has definition of the `forward` method.
    All the children classes should define the self.convs() ModuleList.
    """

    def __init__(self, node_feature_dim, hidden_dim, num_classes=2, edge_attr=True):
        super(ParentHomogeneousGraphLevelGNN, self).__init__()
        self.bns = None
        self.drs = None
        self.convs = None
        self.use_edge_attr = edge_attr
        self.node_feature_dim = node_feature_dim

        self.use_sip = get_configurations_dtype_boolean(section='TRAINING',
                                                        key='USE_SIP')
        self.global_pool_aggr = deterministic_global_add_pool  # global_add_pool
        self.is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP',
                                                                      key='PERFORM_NODE_LEVEL_PREDICTION')

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

    def forward(self, entire_graph_data):
        aggregated_node_features = []
        all_labels = []
        for data, label in entire_graph_data:
            data, label = data.to(device), label.to(device)
            all_labels.append(label)
            # We go through a loop and process each of the smaller graphs.
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
            if self.training and get_configurations_dtype_boolean(section='TRAINING', key='NODE_TRANS'):
                # Apply node augmentation
                coord, num_lesions = x[:, -3:], x.shape[0]
                t = get_configurations_dtype_float(section='TRAINING', key='MAX_TRANS')
                translated_coord = coord.new_empty((num_lesions, 3)).uniform_(-t, t)
                x[:, -3:] += translated_coord
            if self.is_node_level_dataset:
                x = self.update_node_feat(x)
                x1 = F.relu(x)
                x1 = self.dr(x1)
                vol_regr = torch.log(1 + torch.exp(self.regress_lesion_volume(x1)))

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
            if self.global_pool_aggr == deterministic_global_add_pool:
                pooling_out = self.global_pool_aggr(x, None)
            else:
                pooling_out = self.global_pool_aggr(x)
            aggregated_node_features.append(pooling_out)
            # We can now delete the object and free up some memory
            del data, label
        return torch.cat(aggregated_node_features, dim=0), torch.cat(all_labels, dim=0)

    def __repr__(self):
        return f"{self.__class__.__name__}. Abstract parent class with only the forward method."

    def get_full_des(self):
        return super().__repr__()


class SAGEHomConv(ParentHomogeneousGraphLevelGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(SAGEHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes, edge_attr=False)
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


class GoGModel(nn.Module):
    def __init__(self, hidden_dim, node_feature_dim, num_classes=2, total_number_of_gnn_layers=2):
        super(GoGModel, self).__init__()
        self.individual_graph_proc = SAGEHomConv(hidden_dim=hidden_dim, node_feature_dim=node_feature_dim,
                                                 total_number_of_gnn_layers=total_number_of_gnn_layers)
        self.node_processor = DynamicEdgeConv(
            nn=nn.Sequential(
                nn.Linear(in_features=2 * hidden_dim // 8, out_features=hidden_dim // 8),
                nn.ReLU(),
                nn.Dropout()
            ),
            aggr='sum',
            k=5
        )
        self.lin1 = nn.Linear(hidden_dim // 8, hidden_dim // 16)
        self.lin2 = nn.Linear(hidden_dim // 16, num_classes)

    def reset_parameters(self):
        self.individual_graph_proc.reset_parameters()
        for child in self.children():
            if hasattr(child, 'reset_parameters'):
                child.reset_parameters()

    def forward(self, data):
        # Pass through the graph processor module
        node_feat, node_labels = self.individual_graph_proc(data)
        x = F.relu(self.lin1(node_feat))
        logit = self.lin2(x)
        return logit, node_labels

    def get_full_des(self):
        return f"GoG model with {self.individual_graph_proc.get_full_des()}"

    def __repr__(self):
        return f"GoG with hidden_dim {self.lin1.in_features}"


if __name__ == '__main__':
    model = GoGModel(hidden_dim=32, node_feature_dim=772, num_classes=2, total_number_of_gnn_layers=2)
    dataset = GoGDataset()
    model.to(device)
    print(model)
    print(model(dataset[0])[0].shape)
    print(model(dataset[0])[1].shape)
