import os
import pickle
import sys
from collections import defaultdict

import networkx as nx
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch_geometric.transforms import ToSparseTensor

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR, get_configurations_dtype_boolean, \
    device
from graph_models.model_factory import get_model
from utils.dataset_util import get_patient_name_from_dataset


def predict_on_graph(graph, model):
    with torch.no_grad():
        out = model(graph)
        return out


def predict_with_grad_on_graph(graph, model):
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


def get_test_split_indices():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
    return test_indices


def plot_graph_with_pos(out_file, graph_data, lesion_mask=None):
    # Trying with a fixed size plot
    plt.axis([0, 160, 0, 160])
    plt.axis('off')
    # Now creating the graph
    G = nx.Graph()
    if lesion_mask is None:
        # Remove the self loops
        # graph_data.edge_index, graph_data.edge_attr = remove_self_loops(graph_data.edge_index, graph_data.edge_attr)
        nodes_to_plot = graph_data.x
    else:
        nodes_to_plot = graph_data.x[lesion_mask]
    # Now we can go ahead and plot the input
    for idx, node in enumerate(nodes_to_plot):
        # 128 was the fixed scaling factor we used for the coordinates while generating our graph
        spatial_loc = (nodes_to_plot[idx][-3:].cpu().numpy() * 128).tolist()
        # We can only use 2 coordinates for spatial layout.
        G.add_node(idx, pos=(spatial_loc[0], spatial_loc[1], spatial_loc[2]))
    # Also add the edges
    # for (src, dst), w in zip(graph_data.edge_index.cpu().t().tolist(), graph_data.edge_attr.cpu().tolist()):
    #     G.add_edge(src, dst, alpha=w)
    pos = nx.get_node_attributes(G, 'pos')
    print("The node locations are")
    print(pos)
    nx.draw(G, pos)
    plt.savefig(out_file, dpi=250)
    # plt.show()


def plot_graph(graph_data, out_file):
    plot_graph_with_pos(out_file, graph_data)
    # nx_graph = to_networkx_fail_safe(data=graph_data)
    # plot_3d_graph(edge_list=nx_graph.edges(), m_graph=nx_graph,
    #               scan_to_patients=graph_data.graph_metadata.scan_to_patients,
    #               out_file=out_file)
    # plt.show()


def plot_masked_graph(graph, model_output, classification_result, patient_name, plot_save_dir):
    selected_nodes = model_output.get('selected_nodes', None)
    if selected_nodes is None:
        print("We have not used SIP layer. Thus, all nodes are retained. Hence, such a visualization is useless.")
    lesion_mask = torch.zeros(graph.x.shape[0], dtype=torch.bool)
    lesion_mask[selected_nodes] = True
    viz_name = f"graph_{classification_result}_{patient_name}_with_{graph.x.shape[0]}_nodes"
    plot_graph_with_pos(out_file=os.path.join(plot_save_dir, f'original_{viz_name}.png'), graph_data=graph,
                        lesion_mask=None)
    plot_graph_with_pos(out_file=os.path.join(plot_save_dir, f'subgraph_{viz_name}.png'), graph_data=graph,
                        lesion_mask=lesion_mask)


def explain_model_prediction(retention_ratio):
    hidden = 64
    num_layers = 2
    model_type = 'gcn'
    dataset = load_dataset(model_type)
    print(dataset)
    # Let us focus on the large graphs for now.
    # Our model is struggling for large graphs.
    # Perhaps, it makes sense to see what it predicts and if that is something that would make sense.
    auxiliary_string = f"mixup_{get_configurations_dtype_boolean(section='TRAINING', key='USE_MIXUP')}_sip_" \
                       f"{get_configurations_dtype_boolean(section='TRAINING', key='USE_SIP')}"
    correct_pred, dataset_idx = 0, 0
    while correct_pred < 20:
        sample_graph_data, _, graph_label = dataset[dataset_idx]

        dataset_idx += 1
        # Since this is a large graph, we would process it to check the visualizations
        num_nodes = sample_graph_data.x.shape[0]
        # if not num_nodes <= 5:
        #     continue
        model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                          sample_graph_data=sample_graph_data, retention_ratio=retention_ratio)
        # Get correct model weights so that there is no data leakage
        fold = get_fold_from_index(graph_idx=dataset_idx)
        # Let us load the weights of the pretrained model.
        # if fold != 0:
        #     continue
        print(f"Loading model for fold {fold}")
        print(model.get_full_des())
        checkpoint_dir = os.path.join(PROJECT_ROOT_DIR,
                                      get_configurations_dtype_string(section='TRAINING',
                                                                      key='LOG_DIR') + auxiliary_string,
                                      f'_layers_{num_layers}_hidden_dim_{hidden}'
                                      )
        print(f"Loading model from {checkpoint_dir}")
        print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_{fold}.pth"))))
        model.to(device)
        model.eval()
        batch = torch.zeros(sample_graph_data.x.shape[0], dtype=int)
        ptr = torch.tensor([0, sample_graph_data.x.shape[0]], dtype=int)
        sample_graph_data.batch = batch
        sample_graph_data.ptr = ptr
        sample_graph_data = sample_graph_data.to(device)
        # Let us check the model prediction here
        predicted = predict_on_graph(graph=sample_graph_data, model=model)
        patient_name = get_patient_name_from_dataset(sample_graph_data)
        print(
            f"For {patient_name} Ground truth is {graph_label} and prediction is {torch.max(predicted['graph_pred'], dim=1)[1]} with logits{predicted['graph_pred']}")
        classification_result = "correct" if torch.max(predicted['graph_pred'], dim=1)[
                                                 1].cpu() == graph_label else "incorrect"
        if classification_result == 'correct':
            correct_pred += 1
        else:
            # As of now, we look at both the kinds of results but we may also decide to skip the wrong ones.
            pass
        plot_save_dir = os.path.join(PROJECT_ROOT_DIR,
                                     get_configurations_dtype_string(section='TRAINING',
                                                                     key='LOG_DIR') + auxiliary_string
                                     )
        os.makedirs(plot_save_dir, exist_ok=True)

        plot_masked_graph(graph=sample_graph_data, classification_result=classification_result,
                          patient_name=patient_name,
                          model_output=predicted, plot_save_dir=plot_save_dir)


def generate_location_hist(plot_all_lesions=False, retention_ratio=0.5):
    hidden = 64
    num_layers = 2
    model_type = 'gcn'
    dataset = load_dataset(model_type)
    print(dataset)

    auxiliary_string = f"mixup_{get_configurations_dtype_boolean(section='TRAINING', key='USE_MIXUP')}_sip_" \
                       f"{get_configurations_dtype_boolean(section='TRAINING', key='USE_SIP')}"
    lesion_loc_dist = defaultdict(int)
    csv_path = get_configurations_dtype_string(section='SETUP', key='RAW_METADATA_CSV')
    all_patients_df = pd.read_csv(csv_path)
    for idx in range(len(dataset)):
        sample_graph_data, _, graph_label = dataset[idx]

        model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                          sample_graph_data=sample_graph_data, retention_ratio=retention_ratio)
        # Get correct model weights so that there is no data leakage
        fold = get_fold_from_index(graph_idx=idx)
        # Let us load the weights of the pretrained model.
        print(f"Loading model for fold {fold}")
        print(model.get_full_des())
        checkpoint_dir = os.path.join(PROJECT_ROOT_DIR,
                                      get_configurations_dtype_string(section='TRAINING',
                                                                      key='LOG_DIR') + auxiliary_string,
                                      f'_layers_{num_layers}_hidden_dim_{hidden}'
                                      )
        print(f"Loading model from {checkpoint_dir}")
        print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_loss_{fold}.pth"))))
        model.to(device)
        model.eval()
        batch = torch.zeros(sample_graph_data.x.shape[0], dtype=int)
        ptr = torch.tensor([0, sample_graph_data.x.shape[0]], dtype=int)
        sample_graph_data.batch = batch
        sample_graph_data.ptr = ptr
        sample_graph_data = sample_graph_data.to(device)
        # Let us check the model prediction here
        model_output = predict_on_graph(graph=sample_graph_data, model=model)
        patient_name = get_patient_name_from_dataset(sample_graph_data)
        selected_nodes = model_output.get('selected_nodes', None)
        if selected_nodes is None and not plot_all_lesions:
            print("We have not used SIP layer. Thus, all nodes are retained. Hence, such an visualization is useless.")
            sys.exit(0)
        if not plot_all_lesions:
            lesion_mask = torch.zeros(sample_graph_data.x.shape[0], dtype=torch.bool)
            lesion_mask[selected_nodes] = True
        else:
            lesion_mask = torch.ones(sample_graph_data.x.shape[0], dtype=torch.bool)
        spm_selected_nodes = sample_graph_data.x[lesion_mask]
        for idx, node in enumerate(spm_selected_nodes):
            # 128 was the fixed scaling factor we used for the coordinates while generating our graph
            spatial_loc = (spm_selected_nodes[idx][-3:].cpu().numpy() * 128).tolist()
            brain_region = int(all_patients_df[
                                   (patient_name == all_patients_df['Patient']) &
                                   (round(all_patients_df['x']) == round(spatial_loc[0])) &
                                   (round(all_patients_df['y']) == round(spatial_loc[1])) &
                                   (round(all_patients_df['z']) == round(spatial_loc[2]))]['LesionLocation'].item())
            lesion_loc_dist[brain_region] += 1
    # Now we can plot the final histogram results
    # We first sort the counts based on the keys
    lesion_loc_dist = dict(sorted(lesion_loc_dist.items(), key=lambda x: x[0]))
    # x_pos = 1, 2, 3, 4
    print(lesion_loc_dist)


def load_dataset(model_type):
    if model_type in ['gat']:
        print(f"{model_type} does not support edge_attr so dropping edge_attr information")
        dataset = get_dataset(transform=ToSparseTensor(attr=None))
    else:
        print("edge_attr supported. Including edge_attr information")
        dataset = get_dataset(transform=ToSparseTensor(attr='edge_attr'))
    return dataset


if __name__ == '__main__':
    explain_model_prediction(retention_ratio=0.1)
    # generate_location_hist(plot_all_lesions=False, retention_ratio=0.01)
