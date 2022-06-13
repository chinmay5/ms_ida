import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool, GCNConv, HeteroConv, GATv2Conv, SAGEConv, GraphConv

from dataset.PatientDataset import HeterogeneousPatientDataset


class ParentHeterogeneousGNN(nn.Module):
    """
    This is the parent class for all the Homogeneous convolution graphs.
    The parent class has definition of the `forward` method.
    All the children classes should define the self.convs() ModuleList.
    """

    def __init__(self, hidden_dim, num_classes=2):
        super(ParentHeterogeneousGNN, self).__init__()
        self.lin1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.lin2 = nn.Linear(hidden_dim // 2, num_classes)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x_dict, edge_index_dict, batch = data.x_dict, data.edge_index_dict, data.batch_dict
        for idx, conv in enumerate(self.convs):
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x = global_mean_pool(x_dict['lesion'], batch['lesion'])
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}. Abstract parent class with only the forward method."


class GCNHetConv(ParentHeterogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, hetero_dataset_sample, num_classes=2):
        super(GCNHetConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [HeteroConv({
            key: GCNConv(in_channels=node_feature_dim, out_channels=hidden_dim) for key in
            hetero_dataset_sample.edge_index_dict.keys()
        }, aggr='sum')]
        # Now, how are we going to generate all these heterogeneous conv layers?
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(HeteroConv({
                key: GCNConv(in_channels=hidden_dim, out_channels=hidden_dim) for key in
                hetero_dataset_sample.edge_index_dict.keys()
            }, aggr='sum'))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses GCN Convolutions."


class GATHetConv(ParentHeterogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, hetero_dataset_sample, num_classes=2):
        super(GATHetConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [HeteroConv({
            key: GATv2Conv(in_channels=node_feature_dim, out_channels=hidden_dim) for key in
            hetero_dataset_sample.edge_index_dict.keys()
        }, aggr='sum')]
        # Now, how are we going to generate all these heterogeneous conv layers?
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(HeteroConv({
                key: GATv2Conv(in_channels=hidden_dim, out_channels=hidden_dim) for key in
                hetero_dataset_sample.edge_index_dict.keys()
            }, aggr='sum'))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Hetero GAT Convolutions."


class SAGEHetConv(ParentHeterogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, hetero_dataset_sample, num_classes=2):
        super(SAGEHetConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [HeteroConv({
            key: SAGEConv(in_channels=node_feature_dim, out_channels=hidden_dim) for key in
            hetero_dataset_sample.edge_index_dict.keys()
        }, aggr='sum')]
        # Now, how are we going to generate all these heterogeneous conv layers?
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(HeteroConv({
                key: SAGEConv(in_channels=hidden_dim, out_channels=hidden_dim) for key in
                hetero_dataset_sample.edge_index_dict.keys()
            }, aggr='sum'))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Hetero SAGE Convolutions."


class GraphConvHetConv(ParentHeterogeneousGNN):
    def __init__(self, hidden_dim, total_number_of_gnn_layers, node_feature_dim, hetero_dataset_sample, num_classes=2):
        super(GraphConvHetConv, self).__init__(hidden_dim=hidden_dim, num_classes=num_classes)
        convs = [HeteroConv({
            key: GraphConv(in_channels=node_feature_dim, out_channels=hidden_dim) for key in
            hetero_dataset_sample.edge_index_dict.keys()
        }, aggr='sum')]
        # Now, how are we going to generate all these heterogeneous conv layers?
        for _ in range(total_number_of_gnn_layers - 1):
            convs.append(HeteroConv({
                key: GraphConv(in_channels=hidden_dim, out_channels=hidden_dim) for key in
                hetero_dataset_sample.edge_index_dict.keys()
            }, aggr='sum'))
        self.convs = nn.ModuleList(convs)

    def __repr__(self):
        return f"{self.__class__.__name__}. Uses Hetero GraphConv Convolutions."


if __name__ == '__main__':
    # gcn_conv = GCNHomConv(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=512, num_classes=2)
    dataset = HeterogeneousPatientDataset()
    gcn_conv = GCNHetConv(hidden_dim=128, total_number_of_gnn_layers=4, node_feature_dim=512, num_classes=2,
                          hetero_dataset_sample=dataset[0][0])

    dataset = HeterogeneousPatientDataset()
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    data, label = next(iter(dataloader))
    output = gcn_conv(data)
    print(output.shape)
