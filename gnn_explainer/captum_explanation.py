import os

import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients
from torch_geometric.nn import Explainer, to_captum

# Node and edge explainability
# ============================
from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_boolean, PROJECT_ROOT_DIR, get_configurations_dtype_string, \
    get_configurations_dtype_int
from graph_models.model_factory import get_model
from utils.dataset_util import get_patient_name_from_dataset
from utils.explainability_utils import get_fold_from_index, predict_on_graph, CaptumCompatibleModelWrapper, \
    ExplainabilityViz


def get_model_and_dataset(dataset, hidden, idx, model_type, num_layers):
    is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION')
    if is_node_level_dataset:
        sample_graph_data, _, graph_label = dataset[idx]
    else:
        sample_graph_data, graph_label = dataset[idx]
    model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                      sample_graph_data=sample_graph_data)
    # Get correct model weights so that there is no data leakage
    fold = get_fold_from_index(graph_idx=idx)
    # Let us load the weights of the pretrained model.
    print(f"Loading model for fold {fold}")
    checkpoint_dir = os.path.join(PROJECT_ROOT_DIR,
                                  get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'),
                                  f'_layers_{num_layers}_hidden_dim_{hidden}'
                                  )
    print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_{fold}.pth"))))
    model.eval()
    return model, sample_graph_data, graph_label, is_node_level_dataset


def execute_model():
    hidden = 256
    num_layers = 2
    model_type = 'gcn'
    dataset = get_dataset()
    print(dataset)

    diff_graph_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')
    num_large_graphs_processed, dataset_idx = 0, 0
    while num_large_graphs_processed <= 10:
        model, graph, graph_label, is_node_level_dataset = get_model_and_dataset(dataset, hidden, dataset_idx, model_type,
                                                                                 num_layers)
        dataset_idx += 1
        if graph.x.shape[0] <= diff_graph_threshold:
            # We want to process only the large graphs as of now.
            continue
        # Since this is a large graph, we would process it to check the visualizations
        num_large_graphs_processed += 1
        graph_fur_node_explanability, graph_fur_node_and_edge_explanability = graph.clone(), graph.clone()
        # It would enable us to create better visualizations.
        explainability_viz = ExplainabilityViz(graph_data=graph)
        print(model.get_full_des())
        model_wrapper = CaptumCompatibleModelWrapper(nn_model=model)
        patient_name = get_patient_name_from_dataset(graph)

        # Let us check the model prediction here
        predicted = predict_on_graph(graph=graph, is_node_level_dataset=is_node_level_dataset, model=model)
        classification_result = "correct" if torch.max(predicted, dim=1)[1] == graph_label else "incorrect"
        viz_name = f"graph_{classification_result}_{patient_name}_with_{graph.x.shape[0]}_nodes"
        target = torch.max(predicted, dim=1)[1]
        print(
            f"For {patient_name} Ground truth is {graph_label} and prediction is {target} with logits{predicted}")

        # Edge explainability
        captum_model = to_captum(model=model_wrapper, mask_type='edge')
        edge_mask = torch.ones(graph.num_edges, requires_grad=True, device=graph.x.device)

        ig = IntegratedGradients(captum_model)
        ig_attr_edge = ig.attribute(edge_mask.unsqueeze(0), target=target,
                                    additional_forward_args=(graph.x, graph.edge_index),
                                    internal_batch_size=1)

        # Scale attributions to [0, 1]:
        ig_attr_edge = ig_attr_edge.squeeze(0).abs()
        ig_attr_edge /= ig_attr_edge.max()

        # Visualize absolute values of attributions:
        explainer = Explainer(model_wrapper)
        explainability_viz.visualize_subgraph(-1, graph.edge_index, ig_attr_edge, viz_name=f"{viz_name}_edge_explained",
                                              threshold=0.50, is_captum=True)

        # Node explainability
        # ===================

        captum_model = to_captum(model_wrapper, mask_type='node')

        ig = IntegratedGradients(captum_model)
        ig_attr_node = ig.attribute(graph_fur_node_explanability.x.unsqueeze(0), target=target,
                                    additional_forward_args=(graph_fur_node_explanability.edge_index),
                                    internal_batch_size=1)

        # Scale attributions to [0, 1]:
        ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
        ig_attr_node /= ig_attr_node.max()

        # Visualize absolute values of attributions:
        explainer.visualize_subgraph(None, graph_fur_node_explanability.edge_index, ig_attr_edge,
                                     node_alpha=ig_attr_node)
        plt.show()

        # Node and edge explainability
        # ============================

        captum_model = to_captum(model_wrapper, mask_type='node_and_edge')
        ig = IntegratedGradients(captum_model)

        ig_attr_node, ig_attr_edge = ig.attribute(
            (graph_fur_node_and_edge_explanability.x.unsqueeze(0), edge_mask.unsqueeze(0)), target=target,
            additional_forward_args=(graph_fur_node_and_edge_explanability.edge_index), internal_batch_size=1)
        # internal_batch_size=1)

        # Scale attributions to [0, 1]:
        ig_attr_node = ig_attr_node.squeeze(0).abs().sum(dim=1)
        ig_attr_node /= ig_attr_node.max()
        ig_attr_edge = ig_attr_edge.squeeze(0).abs()
        ig_attr_edge /= ig_attr_edge.max()

        # Visualize absolute values of attributions:
        # Visualize absolute values of attributions:
        explainability_viz.visualize_subgraph(node_idx=-1, edge_index=graph_fur_node_and_edge_explanability.edge_index, edge_mask=ig_attr_edge,
                                              node_alpha=ig_attr_node, viz_name=f"{viz_name}_node_and_edge_explained",
                                              threshold=0.50, is_captum=True)


if __name__ == '__main__':
    execute_model()
