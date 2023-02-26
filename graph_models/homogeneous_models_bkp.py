import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv, GATv2Conv, SAGEConv, GraphConv, SGConv
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import dropout_adj

from dataset.PatientDataset import HomogeneousPatientDataset
from utils.training_utils import count_parameters


class AmalgamatedGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim, num_classes=2, total_number_of_gnn_layers=-1):
        super(AmalgamatedGNN, self).__init__()
        self.conv_base = SAGEConv(in_channels=node_feature_dim, out_channels=hidden_dim)
        self.conv_large_only = SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim)
        self.graph_size_map = nn.Embedding(2, hidden_dim // 8)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 8)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 8)
        self.lin2 = nn.Linear(hidden_dim // 8, num_classes)

    def reset_parameters(self):
        self.conv_large_only.reset_parameters()
        self.conv_base.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_index, _ = dropout_adj(edge_index, p=0.5,
                                    force_undirected=False,
                                    num_nodes=data.num_nodes,
                                    training=self.training)
        data.edge_index = edge_index
        sparse_data = ToSparseTensor()(data)
        x = F.leaky_relu(self.conv_base(x, sparse_data.adj_t), negative_slope=0.2)
        # Pass through the second conv layer only when the graph has more than 3 nodes
        graph_sizes = data.ptr[1:] - data.ptr[:-1]
        batch_indices_for_large_graph = torch.where(graph_sizes >= 40)
        large_graphs = sparse_data[batch_indices_for_large_graph]
        x_copy = x.clone()
        if len(large_graphs) > 0:
            x_copy = x_copy[batch_indices_for_large_graph]
            x_copy = F.leaky_relu(self.conv_large_only(x_copy, sparse_data.adj_t), negative_slope=0.2)
            x[batch_indices_for_large_graph] = x_copy

        x = global_mean_pool(x, batch)
        x = F.leaky_relu(self.bn1(self.lin1(x)), negative_slope=0.2)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


class ParentHomogeneousGNN(nn.Module):
    """
    This is the parent class for all the Homogeneous convolution graphs.
    The parent class has definition of the `forward` method.
    All the children classes should define the self.convs() ModuleList.
    """

    def __init__(self, node_feature_dim, hidden_dim, num_classes=2):
        super(ParentHomogeneousGNN, self).__init__()
        self.bns = None
        self.drs = None
        self.convs = None
        self.graph_size_map = nn.Embedding(2, hidden_dim // 8)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 8)
        self.bn1 = nn.BatchNorm1d(hidden_dim // 8)
        self.bn2 = nn.BatchNorm1d(hidden_dim // 8)
        self.lin2 = nn.Linear(hidden_dim // 4, num_classes)
        self.lin0 = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim // 8),
            nn.ReLU()
        )

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.bn1.reset_parameters()
        self.bn2.reset_parameters()
        self.lin0[0].reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        feature_transforms = self.lin0(x)
        # edge_index, _ = dropout_adj(edge_index, p=0.5,
        #                             force_undirected=False,
        #                             num_nodes=data.num_nodes,
        #                             training=self.training)
        # data.edge_index = edge_index
        sparse_data = ToSparseTensor()(data)
        for idx, conv in enumerate(self.convs):
            # x = F.leaky_relu(conv(x, edge_index), negative_slope=0.2)
            x = F.leaky_relu(conv(x, sparse_data.adj_t), negative_slope=0.2)
            if self.bns is not None:
                x = self.bns[idx](x)
            if self.drs is not None:
                x = self.drs[idx](x)
        x = global_mean_pool(x, batch)
        x = F.leaky_relu(self.bn1(self.lin1(x)), negative_slope=0.2)
        # Now add the information about the total number of nodes present
        x1 = self.graph_size_map((torch.bincount(data.batch).unsqueeze(1)>=10).squeeze(1).to(torch.int))
        # Apply BN so that the values are roughly on the same scale
        x1 = self.bn2(x1)
        x1 = F.leaky_relu(x1)
        # Adding the values together
        # x = x + x1
        x1 = torch.sigmoid(x1) * global_mean_pool(feature_transforms, batch)
        x = torch.cat((x, x1), dim=1)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}. Abstract parent class with only the forward method."


class GCNHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GCNHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes)
        convs = [GCNConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GCN Convolutions."


class GATHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GATHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes)
        convs = [GATv2Conv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(GCNConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GAT Convolutions."


class SAGEHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(SAGEHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                          num_classes=num_classes)
        convs = [SAGEConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses SAGE Convolutions."


class GraphConvHomConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(GraphConvHomConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                               num_classes=num_classes)
        convs = [GraphConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(GraphConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GraphConv Convolutions."


class SimpleConv(ParentHomogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(SimpleConv, self).__init__(node_feature_dim=node_feature_dim, hidden_dim=hidden_dim,
                                         num_classes=num_classes)
        convs = [SGConv(in_channels=node_feature_dim, out_channels=hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            convs.append(SGConv(in_channels=hidden_dim, out_channels=hidden_dim))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Simple Graph Convolutions."


class LinearModel(nn.Module):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, num_classes=2):
        super(LinearModel, self).__init__()
        # self.lin0 = nn.Sequential(
        #     nn.Linear(node_feature_dim, node_feature_dim),
        #     nn.ReLU()
        # )
        self.hidden_dim = hidden_dim
        mlp = [nn.Linear(node_feature_dim, hidden_dim)]
        for _ in range(total_number_of_gnn_layers):
            mlp.append(nn.Linear(hidden_dim, hidden_dim))
        self.mlp_layers = nn.ModuleList(mlp)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.fc2 = nn.Linear(hidden_dim // 4, num_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # x = self.lin0(x)
        for mlp in self.mlp_layers:
            x = F.relu(mlp(x))
            x = F.dropout(x, p=0.5, training=self.training)
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
        # self.lin0[0].reset_parameters()

    def __repr__(self):
        return f"Linear model with {self.hidden_dim}"


if __name__ == '__main__':
    gcn_conv = SAGEHomConv(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=768, num_classes=2)
    # gcn_conv = LinearModel(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=512, num_classes=2)
    gcn_conv.reset_parameters()
    dataset = HomogeneousPatientDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    data, label = next(iter(dataloader))
    output = gcn_conv(data)
    print(output.shape)
    print(f"Total number of parameters = {count_parameters(gcn_conv) / 1e6}M")
