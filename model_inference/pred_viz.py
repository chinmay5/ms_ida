import networkx as nx
import os
import pickle
import torch
from matplotlib import pyplot as plt
from torch_geometric.transforms import ToSparseTensor
from torch_geometric.utils import remove_self_loops

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR, get_configurations_dtype_boolean, \
    get_configurations_dtype_int, device
from graph_models.model_factory import get_model
from utils.dataset_util import get_patient_name_from_dataset
from utils.explainability_utils import predict_on_graph, get_fold_from_index
from utils.viz_utils import to_networkx_fail_safe, plot_3d_graph


def get_test_split_indices():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
    return test_indices


def plot_graph_with_pos(graph_data):
    G = nx.Graph()
    # Remove the self loops
    graph_data.edge_index, graph_data.edge_attr = remove_self_loops(graph_data.edge_index, graph_data.edge_attr)
    for idx, node in enumerate(graph_data.x):
        spatial_loc = graph_data.x[idx][-3:].cpu().numpy().tolist()
        # We can only use 2 coordinates for spatial layout.
        G.add_node(idx, pos=(spatial_loc[0], spatial_loc[1]))
    # Also add the edges
    for (src, dst), w in zip(graph_data.edge_index.cpu().t().tolist(), graph_data.edge_attr.cpu().tolist()):
        G.add_edge(src, dst, alpha=w)
    pos = nx.get_node_attributes(G, 'pos')
    nx.draw(G, pos)
    plt.show()


def plot_graph(graph_data, out_file):
    plot_graph_with_pos(graph_data)
    # nx_graph = to_networkx_fail_safe(data=graph_data)
    # plot_3d_graph(edge_list=nx_graph.edges(), m_graph=nx_graph,
    #               scan_to_patients=graph_data.graph_metadata.scan_to_patients,
    #               out_file=out_file)
    # plt.show()


def plot_masked_graph(graph, model_output, classification_result, patient_name):
    weights = model_output['weight_coeff']
    lesion_mask = weights > 0.5
    if lesion_mask.sum() == weights.shape[0]:
        return
    sub_graph = graph.subgraph(lesion_mask.squeeze())
    viz_name = f"graph_{classification_result}_{patient_name}_with_{graph.x.shape[0]}_nodes"
    plot_graph(out_file=f'original_{viz_name}.html', graph_data=graph)
    viz_name = f"graph_{classification_result}_{patient_name}_with_{sub_graph.x.shape[0]}_nodes"
    plot_graph(out_file=f'subgraph_{viz_name}.html', graph_data=sub_graph)


def explain_model_prediction():
    hidden = 64
    num_layers = 2
    model_type = 'sage'
    dataset = get_dataset()
    print(dataset)
    # Let us focus on the large graphs for now.
    # Our model is struggling for large graphs.
    # Perhaps, it makes sense to see what it predicts and if that is something that would make sense.
    diff_graph_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')
    auxiliary_string = f"mixup_{get_configurations_dtype_boolean(section='TRAINING', key='USE_MIXUP')}_sip_" \
                       f"{get_configurations_dtype_boolean(section='TRAINING', key='USE_SIP')}"
    num_large_graphs_processed, dataset_idx = 0, 0
    while num_large_graphs_processed <= 32:
        is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION')
        if is_node_level_dataset:
            sample_graph_data, _, graph_label = dataset[dataset_idx]
        else:
            sample_graph_data, graph_label = dataset[dataset_idx]
        dataset_idx += 1
        if sample_graph_data.x.shape[0] <= diff_graph_threshold:
            # We want to process only the large graphs as of now.
            continue
        # Since this is a large graph, we would process it to check the visualizations
        num_large_graphs_processed += 1
        model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                          sample_graph_data=sample_graph_data)
        # Get correct model weights so that there is no data leakage
        fold = get_fold_from_index(graph_idx=dataset_idx)
        # Let us load the weights of the pretrained model.
        print(f"Loading model for fold {fold}")
        checkpoint_dir = os.path.join(PROJECT_ROOT_DIR,
                                      get_configurations_dtype_string(section='TRAINING', key='LOG_DIR') + auxiliary_string,
                                      f'_layers_{num_layers}_hidden_dim_{hidden}'
                                      )
        print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_{fold}.pth"))))
        model.to(device)
        model.eval()
        batch = torch.zeros(sample_graph_data.x.shape[0], dtype=int)
        ptr = torch.tensor([0, sample_graph_data.x.shape[0]], dtype=int)
        sample_graph_data.batch = batch
        sample_graph_data.ptr = ptr
        sample_graph_data = sample_graph_data.to(device)
        # Let us check the model prediction here
        predicted = predict_on_graph(graph=sample_graph_data, is_node_level_dataset=is_node_level_dataset, model=model)
        patient_name = get_patient_name_from_dataset(sample_graph_data)
        print(
            f"For {patient_name} Ground truth is {graph_label} and prediction is {torch.max(predicted['graph_pred'], dim=1)[1]} with logits{predicted['graph_pred']}")
        classification_result = "correct" if torch.max(predicted['graph_pred'], dim=1)[
                                                 1].cpu() == graph_label else "incorrect"

        plot_masked_graph(graph=sample_graph_data, classification_result=classification_result,
                          patient_name=patient_name,
                          model_output=predicted)


if __name__ == '__main__':
    explain_model_prediction()
