import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv, GATv2Conv, SAGEConv, GraphConv
from torch_geometric.utils import dropout_adj

from dataset.PatientDataset import HomogeneousPatientDataset


class ParentHomogeneousGNN(nn.Module):
    """
    This is the parent class for all the Homogeneous convolution graphs.
    The parent class has definition of the `forward` method.
    All the children classes should define the self.convs() ModuleList.
    """

    def __init__(self, hidden_dim, num_classes=2):
        super(ParentHomogeneousGNN, self).__init__()
        self.bns = None
        self.drs = None
        self.convs = None
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = dropout_adj(edge_index, p=0.5,
                                    force_undirected=True,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        for idx, conv in enumerate(self.convs):
            x = F.leaky_relu(conv(x, edge_index), negative_slope=0.2)
            if self.bns is not None:
                x = self.bns[idx](x)
            if self.drs is not None:
                x = self.drs[idx](x)
        x = global_mean_pool(x, batch)
        x = F.leaky_relu(self.lin1(x), negative_slope=0.2)
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}. Abstract parent class with only the forward method."


class GCNHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GCNHomConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [GCNConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GCN Convolutions."


class GATHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GATHomConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [GATv2Conv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GAT Convolutions."


class SAGEHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(SAGEHomConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [SAGEConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses SAGE Convolutions."


class GraphConvHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GraphConvHomConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [GraphConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(GraphConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GraphConv Convolutions."


class LinearModel(nn.Module):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(LinearModel, self).__init__()
        mlp = [nn.Linear(node_feature_dim, hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp_layers = nn.ModuleList(mlp)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for mlp in self.mlp_layers:
            x = F.relu(mlp(x))
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.fc2(x))
        return x

    def reset_parameters(self):
        for mlp in self.mlp_layers:
            mlp.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()


if __name__ == '__main__':
    gcn_conv = GCNHomConv(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=768, num_classes=2)
    # gcn_conv = LinearModel(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=512, num_classes=2)
    dataset = HomogeneousPatientDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    data, label = next(iter(dataloader))
    output = gcn_conv(data)
    print(output.shape)
