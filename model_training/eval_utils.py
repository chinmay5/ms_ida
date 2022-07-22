from collections import defaultdict

import os
import pickle
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix

from environment_setup import get_configurations_dtype_int
from utils.training_utils import LabelEncoder, CustomDictKey
from utils.viz_utils import plot_bar_plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

smallness_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')
# We decide for three possible graph sizes
graph_size_small = CustomDictKey(key_name=f"less than {smallness_threshold}", key_iden=0)
graph_size_large = CustomDictKey(key_name=f"more than {smallness_threshold}", key_iden=1)


def eval_acc(model, loader):
    model.eval()
    correct = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    return correct / len(loader.dataset)


def eval_acc_with_confusion_matrix(model, loader):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    correct = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        outPRED = torch.cat((outPRED, pred), 0)
        outGT = torch.cat((outGT, labels), 0)
        correct += pred.eq(labels.view(-1)).sum().item()
    confusion_mat = compute_confusion_matrix(gt=outGT, predictions=outPRED, is_prediction=True)
    return correct / len(loader.dataset), confusion_mat


def eval_roc_auc(model, loader, enc, epoch=0, writer=None):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            pred = model(data)
        outPRED = torch.cat((outPRED, pred), 0)
        outGT = torch.cat((outGT, labels), 0)
    predictions = torch.softmax(outPRED, dim=1)
    predictions, target = predictions.cpu().numpy(), outGT.cpu().numpy()
    # Encoder is callable.
    # Hence, we execute callable which returns the self.encoder instance
    target_one_hot = enc().transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions)
    if writer is not None:
        writer.add_scalar('roc', roc_auc_value, global_step=epoch)
    return roc_auc_value


def eval_loss(model, loader, criterion, epoch, writer):
    model.eval()
    # Some information needed for logging on tensorboard
    total_loss = 0
    for idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)
        loss = criterion(out, labels.view(-1)).item()
        total_loss += loss
    avg_val_loss = total_loss / len(loader)
    writer.add_scalar('loss', avg_val_loss, global_step=epoch)
    return avg_val_loss


def decide_graph_category_based_on_size(graph_size):
    if graph_size <= smallness_threshold:
        return graph_size_small
    else:
        return graph_size_large


def eval_graph_len_acc(model, dataset):
    model.eval()
    # A dictionary of the format
    # {
    #   size_a:
    #           [
    #           (tensor([prob_0, prob_1]), gt1),
    #           (tensor([prob_0, prob_1]), gt2),
    #           (tensor([prob_0, prob_1]), gt3),
    #           ]
    #  size_b:
    #           ...
    # }
    size_cm_dict = defaultdict(list)
    correct = 0
    for idx in range(len(dataset)):
        graph, graph_label = dataset[idx]
        # We need to add a dummy batch attribute to our graph.
        batch = torch.zeros(graph.x.shape[0], dtype=int, device=graph.x.device)
        graph.batch = batch
        graph, graph_label = graph.to(device), torch.as_tensor(graph_label).to(device)
        graph_size_categ = decide_graph_category_based_on_size(graph.x.size(0))
        with torch.no_grad():
            model_pred = model(graph)
            pred = model_pred.max(1)[1]
        size_cm_dict[graph_size_categ].append([model_pred, graph_label.item()])
        correct += pred.eq(graph_label.view(-1)).sum().item()
    return correct / len(dataset), size_cm_dict


def _compute_roc_for_graph_size(predictions, gt, enc):
    predictions, gt = predictions.cpu().numpy(), gt.cpu().numpy()
    target_one_hot = enc().transform(gt.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions)
    return roc_auc_value


def compute_confusion_matrix(gt, predictions, is_prediction=False):
    if not is_prediction:
        predicted_label = predictions.max(1)[1]
    else:
        predicted_label = predictions
    gt, predicted_label = gt.cpu().numpy(), predicted_label.cpu().numpy()
    return confusion_matrix(gt, predicted_label)


def plot_results_based_on_graph_size(size_cm_dict, filename_acc, filename_roc, model_type=None, output_dir=None, fold=0,
                                     is_plotting_enabled=True):
    accuracy_dictionary, roc_dictionary, cm_dict = {}, {}, {}
    enc = LabelEncoder()
    for graph_size, model_predictions_list in size_cm_dict.items():
        predictions, gt = torch.concat([x[0] for x in model_predictions_list]), torch.stack(
            [torch.as_tensor(x[1]) for x in model_predictions_list])
        gt = gt.to(predictions.device)
        acc = compute_acc(gt, predictions)
        accuracy_dictionary[graph_size] = acc
        cm = compute_confusion_matrix(gt, predictions)
        cm_dict[graph_size] = cm
        # ROC is not defined in case the gt is all 0s or all 1s.
        # So, we would use the accuracy in these cases to give us an indication
        try:
            roc = _compute_roc_for_graph_size(predictions, gt, enc)
            roc_dictionary[graph_size] = roc
        except ValueError:
            print(f"roc not defined since gt is {gt}")
    if is_plotting_enabled:
        plot_bar_plot(dictionary_to_plot=accuracy_dictionary, y_label='accuracy',
                      title=f'{model_type} accuracy vs. size',
                      filename=filename_acc, output_dir=output_dir)
        plot_bar_plot(dictionary_to_plot=roc_dictionary, y_label='roc', title=f'{model_type} roc vs. size',
                      filename=filename_roc, output_dir=output_dir, color='b')
    if output_dir is not None:
        cm_save_path = os.path.join(output_dir, f'cm{fold}.pkl')
        pickle.dump(cm_dict, open(cm_save_path, 'wb'))
    return accuracy_dictionary, roc_dictionary


def plot_avg_of_dictionary(input_dict, y_label, filename, output_dir, color):
    """

    :param input_dict: A dictionary with string key and a list of values to reduce
    :param y_label: plot label
    :param filename: filename to save the plot
    :param output_dir: directory location for saving plots
    :param color: color of bar plot
    :return: None
    """
    avg_dict = {}
    for key, item_list in input_dict.items():
        avg_dict[key] = sum(item_list) / len(item_list)
    plot_bar_plot(dictionary_to_plot=avg_dict, y_label=y_label, title=f'{filename} {y_label} vs. size',
                  filename=filename, output_dir=output_dir, color=color)


def compute_acc(gt, predictions):
    predicted_label = predictions.max(1)[1]
    acc = predicted_label.eq(gt.view(-1)).sum().item() / predictions.shape[0]
    return acc
