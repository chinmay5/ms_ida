import os
import pickle

import numpy as np
import matplotlib as mpl
import torch
from matplotlib import pyplot as plt

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR, get_configurations_dtype_boolean, \
    get_configurations_dtype_int
from graph_models.model_factory import get_model
from utils.dataset_util import get_patient_name_from_dataset
from utils.explainability_utils import predict_on_graph, get_fold_from_index

viz_save_rel_path = get_configurations_dtype_string(section='VISUALIZATIONS', key='SAVE_HTML_REL_PATH')
heatmap_save_path = os.path.join(PROJECT_ROOT_DIR, viz_save_rel_path, "heatmaps")
os.makedirs(heatmap_save_path, exist_ok=True)


def get_test_split_indices():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
    return test_indices


def execute_occlusion_method(model, sample_graph_data, target, is_node_level_dataset, viz_name):
    # We will iterate through all the different nodes, keep removing them and record logits
    heatmaps = []
    for idx in range(sample_graph_data.x.shape[0]):
        mask = torch.ones(sample_graph_data.x.shape[0], dtype=torch.bool)
        mask[idx] = False
        sub_graph = sample_graph_data.subgraph(mask)
        logits = predict_on_graph(sub_graph, is_node_level_dataset=is_node_level_dataset, model=model)
        class_specific_logit = logits[0, target]
        heatmaps.append(torch.sigmoid(class_specific_logit))
    heatmaps = torch.stack(heatmaps)
    plt.gca().axes.get_xaxis().set_visible(False)
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    cmap = mpl.cm.inferno
    plt.imshow(heatmaps, norm=norm, cmap=cmap)
    plt.colorbar()
    plt.savefig(os.path.join(heatmap_save_path, viz_name), dpi=250)
    plt.show()


def explain_model_prediction():
    hidden = 256
    num_layers = 2
    model_type = 'gcn'
    dataset = get_dataset()
    print(dataset)
    # Let us focus on the large graphs for now.
    # Our model is struggling for large graphs.
    # Perhaps, it makes sense to see what it predicts and if that is something that would make sense.
    diff_graph_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')
    num_large_graphs_processed, dataset_idx = 0, 0
    while num_large_graphs_processed <= 10:
        is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION')
        if is_node_level_dataset:
            sample_graph_data, _, graph_label = dataset[dataset_idx]
        else:
            sample_graph_data, graph_label = dataset[dataset_idx]
        dataset_idx += 1
        if graph_label == 1 or sample_graph_data.x.shape[0] <= diff_graph_threshold:
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
                                      get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'),
                                      f'_layers_{num_layers}_hidden_dim_{hidden}'
                                      )
        print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_{fold}.pth"))))
        model.eval()
        batch = torch.zeros(sample_graph_data.x.shape[0], dtype=int, device=sample_graph_data.x.device)
        sample_graph_data.batch = batch
        # Let us check the model prediction here
        predicted = predict_on_graph(graph=sample_graph_data, is_node_level_dataset=is_node_level_dataset, model=model)
        patient_name = get_patient_name_from_dataset(sample_graph_data)
        prediction_label = torch.max(predicted, dim=1)[1]
        print(
            f"For {patient_name} Ground truth is {graph_label} and prediction is {prediction_label} with logits{predicted}")
        classification_result = "correct" if prediction_label == graph_label else "incorrect"

        viz_name = f"Occlusion_graph_{classification_result}_{patient_name}_with_{sample_graph_data.x.shape[0]}_nodes"
        execute_occlusion_method(model=model, sample_graph_data=sample_graph_data, target=prediction_label,
                                 is_node_level_dataset=is_node_level_dataset, viz_name=viz_name)


if __name__ == '__main__':
    explain_model_prediction()
