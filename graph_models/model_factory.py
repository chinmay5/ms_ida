from environment_setup import get_configurations_dtype_int, get_configurations_dtype_string, \
    get_configurations_dtype_boolean
from graph_models.heterogeneous_models import GCNHetConv, GATHetConv, GraphConvHetConv, SAGEHetConv
from graph_models.homogeneous_models import LinearModel, GCNHomConv, GATHomConv, SAGEHomConv, GraphConvHomConv


def get_model(model_type, hidden_dim, num_layers, sample_graph_data):
    num_classes = get_configurations_dtype_int(section='TRAINING', key='NUM_CLASSES')
    node_feature_dim = get_configurations_dtype_int(section='TRAINING', key='NODE_FEAT_DIM')
    # model_type = get_configurations_dtype_string(section='TRAINING', key='MODEL_TYPE')
    is_hetero = get_configurations_dtype_boolean(section='TRAINING', key='IS_HETERO')
    model_type = f'het_{model_type}' if is_hetero else model_type
    if model_type == 'linear':
        return LinearModel(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                           node_feature_dim=node_feature_dim, num_classes=num_classes)
    elif model_type == 'gcn':
        return GCNHomConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                          node_feature_dim=node_feature_dim, num_classes=num_classes)
    elif model_type == 'gat':
        return GATHomConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                          node_feature_dim=node_feature_dim, num_classes=num_classes)
    elif model_type == 'sage':
        return SAGEHomConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                          node_feature_dim=node_feature_dim, num_classes=num_classes)
    elif model_type == 'graph_conv':
        return GraphConvHomConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                          node_feature_dim=node_feature_dim, num_classes=num_classes)
    elif model_type == 'het_gcn':
        return GCNHetConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers, node_feature_dim=node_feature_dim, num_classes=num_classes,
                          hetero_dataset_sample=sample_graph_data)
    elif model_type == 'het_gat':
        return GATHetConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers, node_feature_dim=node_feature_dim, num_classes=num_classes,
                          hetero_dataset_sample=sample_graph_data)
    elif model_type == 'het_sage':
        return SAGEHetConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                          node_feature_dim=node_feature_dim, num_classes=num_classes,
                           hetero_dataset_sample=sample_graph_data)
    elif model_type == 'het_graph_conv':
        return GraphConvHetConv(hidden_dim=hidden_dim, total_number_of_gnn_layers=num_layers,
                          node_feature_dim=node_feature_dim, num_classes=num_classes,
                                hetero_dataset_sample=sample_graph_data)