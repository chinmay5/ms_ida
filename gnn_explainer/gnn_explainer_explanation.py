import os
import pickle

import torch
from matplotlib import pyplot as plt
from torch_geometric.nn import GNNExplainer
from torch_geometric.nn.models.explainer import set_masks
from tqdm import tqdm

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR, get_configurations_dtype_boolean, \
    get_configurations_dtype_int
from graph_models.model_factory import get_model
from utils.dataset_util import get_patient_name_from_dataset
from utils.explainability_utils import predict_on_graph, get_fold_from_index, predict_with_grad_on_graph, \
    ExplainabilityViz


class CustomGNNExplainer(GNNExplainer):
    def __init__(self, model, epochs: int = 1000, lr: float = 0.01,
                 num_hops=None, return_type: str = 'log_prob',
                 feat_mask_type: str = 'feature', allow_edge_mask: bool = True,
                 log: bool = True, graph_data=None, **kwargs):
        super(CustomGNNExplainer, self).__init__(model=model, epochs=epochs, lr=lr,
                                                 num_hops=num_hops, return_type=return_type,
                                                 feat_mask_type=feat_mask_type, allow_edge_mask=allow_edge_mask,
                                                 log=log, kwargs=kwargs)
        self.graph_data = graph_data
        self.explainability_viz = ExplainabilityViz(graph_data=graph_data)

    def explain_graph(self, x, edge_index, is_node_level_dataset, **kwargs):

        self.model.eval()
        self._clear_masks()

        # all nodes belong to same graph
        batch = torch.zeros(x.shape[0], dtype=int, device=x.device)
        self.graph_data.batch = batch

        # Get the initial prediction.
        out = predict_on_graph(graph=self.graph_data, is_node_level_dataset=is_node_level_dataset, model=self.model)
        if self.return_type == 'regression':
            prediction = out
        else:
            log_logits = self._to_log_prob(out)
            prediction = log_logits.argmax(dim=-1)

        self._initialize_masks(x, edge_index)
        self.to(x.device)
        if self.allow_edge_mask:
            set_masks(self.model, self.edge_mask, edge_index,
                      apply_sigmoid=True)
            parameters = [self.node_feat_mask, self.edge_mask]
        else:
            parameters = [self.node_feat_mask]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)

        if self.log:  # pragma: no cover
            pbar = tqdm(total=self.epochs)
            pbar.set_description('Explain graph')

        for epoch in range(1, self.epochs + 1):
            optimizer.zero_grad()
            h = x * self.node_feat_mask.sigmoid()
            self.graph_data.x = h
            out = predict_with_grad_on_graph(graph=self.graph_data, is_node_level_dataset=is_node_level_dataset,
                                             model=self.model)
            loss = self.get_loss(out, prediction, None)
            loss.backward()
            optimizer.step()

            if self.log:  # pragma: no cover
                pbar.update(1)

        if self.log:  # pragma: no cover
            pbar.close()

        node_feat_mask = self.node_feat_mask.detach().sigmoid().squeeze()
        if self.allow_edge_mask:
            edge_mask = self.edge_mask.detach().sigmoid()
        else:
            edge_mask = torch.ones(edge_index.size(1))

        self._clear_masks()
        return node_feat_mask, edge_mask


def get_test_split_indices():
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
    return test_indices


def execute_gnn_explainer(model, graph, is_node_level_dataset, viz_name):
    explainer = CustomGNNExplainer(model, epochs=300, return_type='raw', graph_data=graph, lr=0.1)
    node_feat_mask, edge_mask = explainer.explain_graph(graph.x, graph.edge_index,
                                                        is_node_level_dataset=is_node_level_dataset)
    # node_idx -1 is used to explain the entire graph
    explainer.explainability_viz.visualize_subgraph(node_idx=-1, edge_index=graph.edge_index, edge_mask=edge_mask,
                                                    threshold=0.25,
                                                    viz_name=viz_name)
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
        print(
            f"For {patient_name} Ground truth is {graph_label} and prediction is {torch.max(predicted, dim=1)[1]} with logits{predicted}")
        classification_result = "correct" if torch.max(predicted, dim=1)[1] == graph_label else "incorrect"

        viz_name = f"graph_{classification_result}_{patient_name}_with_{sample_graph_data.x.shape[0]}_nodes"
        execute_gnn_explainer(model=model, graph=sample_graph_data,
                              is_node_level_dataset=is_node_level_dataset, viz_name=viz_name)


if __name__ == '__main__':
    explain_model_prediction()
