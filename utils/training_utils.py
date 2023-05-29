import os
import pickle
import random
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

from environment_setup import PROJECT_ROOT_DIR, get_configurations_dtype_boolean, device, \
    get_configurations_dtype_string, get_configurations_dtype_int
from utils.viz_utils import plot_bar_plot

smallness_threshold = get_configurations_dtype_int(section='SETUP', key='DIFF_GRAPH_THRESHOLD')


class LogWriterWrapper(object):
    def __init__(self, summary_writer=None):
        self.summary_writer = summary_writer

    def add_scalar(self, *args, **kwargs):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(*args, **kwargs)

    def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag="default", metadata_header=None):
        if self.summary_writer is not None:
            self.summary_writer.add_embedding(mat, metadata=metadata, label_img=label_img, global_step=global_step,
                                              tag=tag, metadata_header=metadata_header)


class CustomDictKey(object):
    def __init__(self, key_name, key_iden):
        super(CustomDictKey, self).__init__()
        self.key_name = key_name
        self.key_iden = key_iden

    def __eq__(self, other):
        return isinstance(other, CustomDictKey) and \
               other.key_name == self.key_name and \
               other.key_iden == self.key_iden

    def __hash__(self):
        return hash((self.key_name, self.key_iden))

    def __repr__(self):
        return f'{self.key_name} - {self.key_iden}'


class LabelEncoder(object):

    def __init__(self):
        enc = OneHotEncoder()
        possible_labels = np.array([0, 1]).reshape(-1, 1)
        enc.fit(possible_labels)
        self.encoder = enc

    def __call__(self):
        return self.encoder


class RunTimeConfigs(object):
    def __init__(self):
        self.configs = []

    def write_to_disk(self):
        base_log_dir = os.path.join(PROJECT_ROOT_DIR, self.logdir)
        os.makedirs(base_log_dir, exist_ok=True)
        filename = os.path.join(base_log_dir, "configs_for_run.cfg")
        with open(filename, 'w') as configfile:
            for config, value in vars(self):
                configfile.write(f"{config}: {value} \n")


# Ensuring reproducibility
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


def drop_nodes(graph):
    # We can drop 10% of the nodes at random.
    node_mask = torch.rand(graph.num_nodes) >= 0.1
    # We do not prune when there are a very few nodes left.
    if node_mask.sum() <= 4:
        return graph
    sub_graph = graph.subgraph(node_mask)
    return sub_graph


def graph_size_based_sampler(train_dataset):
    per_sample_wt = [0] * len(train_dataset)
    for idx, (graph, label) in enumerate(train_dataset):
        per_sample_wt[idx] = graph.x.size(0)  # Assigning the length weight to our sampler
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=False)
    return weighted_sampler


def balanced_batch_sampler(train_dataset):
    per_sample_wt = [0] * len(train_dataset)
    class_weights = get_class_weights(train_dataset)
    for idx, (graph, label) in enumerate(train_dataset):
        class_weight = class_weights[label]
        per_sample_wt[idx] = class_weight  # Assigning the length weight to our sampler
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=True)
    return weighted_sampler


def get_class_weights(train_dataset):
    # We also need to obtain class weights to ensure we do not have data imbalance issues.
    pos_samples = sum([sample[1] for sample in train_dataset])
    neg_samples = len(train_dataset) - pos_samples
    # Label 0 is negative and label 1 is positive
    if pos_samples > neg_samples:
        class_balance_weights = torch.as_tensor([pos_samples / neg_samples, 1], device=device)
    else:
        class_balance_weights = torch.as_tensor([1, neg_samples / pos_samples], device=device)
    return class_balance_weights


def get_dataset_and_auxiliary_loss(dataset, test_idx, train_idx, val_idx, no_aug):
    is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION')
    print(f"Is node level dataset: {is_node_level_dataset}")

    if no_aug:
        train_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in train_idx]
    else:
        train_dataset = [(dataset[idx.item()][1], dataset[idx.item()][2]) for idx in train_idx]
    test_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in test_idx]
    val_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in val_idx]
    return test_dataset, train_dataset, val_dataset


def create_weighted_sampler(class_weights, dataset):
    # Trying the weighted sampler idea rather than the loss re-weighting
    # We need to assign a weight to each of the samples in our dataset
    per_sample_wt = [0] * len(dataset)
    for idx, (graph, label) in enumerate(dataset):
        cls_wt = class_weights[label]  # Finding the weight associated with our given label
        per_sample_wt[idx] = cls_wt  # Assigning this class-based weight to our sample
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=False)
    return weighted_sampler


def sanity_check(train_indices, val_indices, test_indices):
    per_split_result = []
    for idx in range(len(train_indices)):
        train_set = set(train_indices[idx].numpy().tolist())
        val_set = set(val_indices[idx].numpy().tolist())
        test_set = set(test_indices[idx].numpy().tolist())
        per_split_result.append(
            all([len(train_set.intersection(val_set)) == 0, len(val_set.intersection(test_set)) == 0,
                 len(train_set.intersection(test_set)) == 0]))
    return all(per_split_result)


def k_fold(dataset, folds):
    # We define the splits once and re-use them.
    # This is one way of reducing possible stochasticity.
    k_fold_split_path = get_configurations_dtype_string(section='SETUP', key='K_FOLD_SPLIT_PATH')
    create_fresh_split = True
    if os.path.exists(k_fold_split_path):
        num_folds = pickle.load(open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'rb'))
        if num_folds == folds:
            print("Using a pre-defined k fold split. Done for easy reproducibility.")
            train_indices = pickle.load(open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'rb'))
            val_indices = pickle.load(open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'rb'))
            test_indices = pickle.load(open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'rb'))
            create_fresh_split = False
        else:
            print("Number of folds differ. Creating a fresh train-val-test split")
    # Create a new data split
    if create_fresh_split:
        print("Generating a new k-fold split")
        skf = StratifiedKFold(folds, shuffle=True, random_state=42)

        test_indices, train_indices = [], []
        # https://stackoverflow.com/questions/45516424/sklearn-train-test-split-on-pandas-stratify-by-multiple-columns
        # for _, idx in skf.split(torch.zeros(len(dataset)), dataset.graph_catogory_label.cpu().numpy().tolist()):
        label_and_graph_size = [str(y) + "_" + str(size) for y, size in
                                zip(dataset.y, dataset.graph_catogory_label.cpu().numpy().tolist())]
        for _, idx in skf.split(torch.zeros(len(dataset)), label_and_graph_size):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(folds)]
        # 70-20-10 attempt
        # test_indices = [torch.cat((test_indices[i], test_indices[i - 1])) for i in range(folds)]
        # val_indices = [torch.cat((test_indices[i - 2], test_indices[i - 3])) for i in range(folds)]

        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        # Now, let us go ahead and save these values
        assert sanity_check(train_indices, val_indices, test_indices), "Something wrong with the splits"
        os.makedirs(k_fold_split_path, exist_ok=False)  # Exists ok is not fine here.
        pickle.dump(train_indices, open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'wb'))
        pickle.dump(val_indices, open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'wb'))
        pickle.dump(test_indices, open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
        pickle.dump(folds, open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'wb'))

    return train_indices, val_indices, test_indices


def train_val_loop(criterion, enc, epochs, model, optimizer, roc_auc,
                   train_loader, train_writer, val_loader, val_losses,
                   val_writer, log_dir, fold,
                   scheduler=None):
    best_val_roc = 0
    min_loss = 1e10
    best_model_save_epoch = -1
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train(model=model, optimizer=optimizer, loader=train_loader, criterion=criterion, epoch=epoch,
                           train_writer=train_writer)
        val_loss = eval_loss(model=model, loader=val_loader, criterion=criterion, epoch=epoch, writer=val_writer,
                             plotting_offset=len(val_loader.dataset))
        val_roc = eval_roc_auc(model=model, loader=val_loader, enc=enc, epoch=epoch, writer=val_writer)
        train_roc = eval_roc_auc(model=model, loader=train_loader, enc=enc, epoch=epoch, writer=train_writer)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_roc': val_roc,
            'train_roc': train_roc
        }
        # Appending the results for selecting best model
        val_losses.append(val_loss)
        roc_auc.append(val_roc)

        # if val_roc > best_val_roc:
        #     best_val_roc = val_roc
        #     torch.save(model.state_dict(), os.path.join(log_dir, f"{model}_{fold}.pth"))

        if val_loss < min_loss:
            best_model_save_epoch = epoch
            min_loss = val_loss
            torch.save(model.state_dict(), os.path.join(log_dir, f"{model}_loss_{fold}.pth"))

        if scheduler is not None:
            scheduler.step(val_loss)
        # Our training dataset is a list with data objects.
        # So, in order to get more augmentations, we have to "reload" the list.
        # This ensures the __get_item__ is called repeatedly and thus, we get more augmentations.
    print(f"Best model saved at {best_model_save_epoch}")


def execute_graph_classification_epoch(criterion, data, label, model, optimizer, total_loss, scheduler):
    out = model(data)
    loss = criterion(out['graph_pred'], label.view(-1))
    loss.backward()
    total_loss += loss.item()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return total_loss


def train(model, optimizer, loader, criterion, epoch, train_writer, scheduler=None):
    model.train()
    # Some information needed for logging on tensorboard
    total_loss = 0
    for idx, (data, label) in enumerate(loader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        total_loss = execute_graph_classification_epoch(criterion, data, label, model, optimizer, total_loss,
                                                        scheduler=scheduler)

    avg_train_loss = total_loss / len(loader)
    train_writer.add_scalar('loss', avg_train_loss, global_step=epoch)
    return avg_train_loss


# Eval utils

# We decide for three possible graph sizes
graph_size_small = CustomDictKey(key_name=f"less than {smallness_threshold}", key_iden=0)
graph_size_large = CustomDictKey(key_name=f"more than {smallness_threshold}", key_iden=1)


def min_max_normalize(vector, factor):
    vector = factor * (vector - np.min(vector)) / (np.max(vector) - np.min(vector))
    return vector


def normalize_features(features):
    for ii in range(np.shape(features)[1]):
        features[:, ii] = min_max_normalize(features[:, ii], 1)


def eval_acc(model, loader, criterion_vol_regr, writer=None):
    model.eval()
    correct = 0
    out_feat = torch.FloatTensor().to(device)
    outGT = torch.FloatTensor().to(device)
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)
            pred = out['graph_pred'].max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    if writer is not None:
        return correct / len(loader.dataset), out_feat, outGT
    return correct / len(loader.dataset)


def eval_acc_with_confusion_matrix(model, dataset):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    correct = 0
    for data, labels in dataset:
        batch = torch.zeros(data.x.shape[0], dtype=int, device=data.x.device)
        ptr = torch.tensor([0, data.x.shape[0]], dtype=int, device=data.x.device)
        data.batch = batch
        data.ptr = ptr
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)  # Ignoring the node & regr component for the time being
            pred = out['graph_pred'].max(1)[1]
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, labels), 0)
        # correct += balanced_accuracy_score(labels.view(-1).cpu().numpy(), pred.cpu().numpy())
        correct += pred.eq(labels.view(-1)).sum().item()
    confusion_mat = compute_confusion_matrix(gt=outGT, predictions=outPRED, is_prediction=True)
    return correct / len(dataset), confusion_mat


@torch.no_grad()
def eval_roc_auc(model, loader, enc, epoch=0, writer=None):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            out = model(data)  # Ignoring the node & regr component for the time being
        outPRED = torch.cat((outPRED, out['graph_pred']), 0)
        outGT = torch.cat((outGT, labels), 0)
    predictions = torch.softmax(outPRED, dim=1)
    predictions, target = predictions.cpu().numpy(), outGT.cpu().numpy()
    # Encoder is callable.
    # Hence, we execute callable which returns the self.encoder instance
    target_one_hot = enc().transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    if writer is not None:
        writer.add_scalar('roc', roc_auc_value, global_step=epoch)
    return roc_auc_value


@torch.no_grad()
def eval_roc_auc_with_prec_recall_acc_and_f1(model, loader, enc, epoch=0, writer=None):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.cuda.amp.autocast():
            out = model(data)  # Ignoring the node & regr component for the time being
        outPRED = torch.cat((outPRED, out['graph_pred']), 0)
        outGT = torch.cat((outGT, labels), 0)
    predictions = torch.softmax(outPRED, dim=1)
    predictions, target = predictions.cpu().numpy(), outGT.cpu().numpy()
    # Encoder is callable.
    # Hence, we execute callable which returns the self.encoder instance
    target_one_hot = enc().transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    if writer is not None:
        writer.add_scalar('roc', roc_auc_value, global_step=epoch)
    # Computing the extra metric
    prec, recall, acc, f1 = compute_precision_recall_acc_and_f1(y_true=target, y_pred=predictions)
    return roc_auc_value, prec, recall, acc, f1


def compute_precision_recall_acc_and_f1(y_true, y_pred, threshold=0.675):
    """
    :param y_true: GT label associated with the given example
    :param y_pred: The prediction logits made by the model
    :return: specificity, sensitivity, accuracy, f1
    """
    # y_pred = np.argmax(y_pred, axis=1)
    y_pred = y_pred[:, 1] >= threshold
    cm = confusion_matrix(y_true, y_pred)

    # Extract true negatives, false positives, false negatives, and true positives from the confusion matrix
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Compute F1 score
    f1 = f1_score(y_true, y_pred)
    return precision, recall, accuracy, f1


def eval_loss(model, loader, criterion, epoch, writer, plotting_offset=-1):
    model.eval()
    # Some information needed for logging on tensorboard
    total_loss = 0
    nodes_on = []
    for idx, (data, labels) in enumerate(loader):
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            out = model(data)
            loss = criterion(out['graph_pred'], labels.view(-1)).item()
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
        ptr = torch.tensor([0, graph.x.shape[0]], dtype=int, device=graph.x.device)
        graph.batch = batch
        graph.ptr = ptr
        graph, graph_label = graph.to(device), graph_label.to(device)
        graph_size_categ = decide_graph_category_based_on_size(graph.x.size(0))
        with torch.no_grad():
            out = model(graph)  # Ignoring the node & regr component for the time being
            pred = out['graph_pred'].max(1)[1]
        size_cm_dict[graph_size_categ].append([out['graph_pred'], graph_label.item()])
        correct += pred.item() == graph_label.item()
    return correct / len(dataset), size_cm_dict


def _compute_roc_for_graph_size(predictions, gt, enc):
    predictions, gt = predictions.cpu().numpy(), gt.cpu().numpy()
    target_one_hot = enc().transform(gt.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    return roc_auc_value


def compute_confusion_matrix(gt, predictions, is_prediction=False):
    if not is_prediction:
        predicted_label = predictions.max(1)[1]
    else:
        predicted_label = predictions
    gt, predicted_label = gt.cpu().numpy(), predicted_label.cpu().numpy()
    return confusion_matrix(gt, predicted_label)


def plot_results_based_on_graph_size(size_cm_dict, filename_acc, filename_roc, model_type=None, output_dir=None, fold=0,
                                     is_plotting_enabled=True, split_acc_based_on_labels=False):
    accuracy_dictionary, roc_dictionary, cm_dict = {}, {}, {}
    enc = LabelEncoder()
    skip_this_round = False
    for graph_size, model_predictions_list in size_cm_dict.items():
        predictions, gt = torch.concat([x[0] for x in model_predictions_list]), torch.stack(
            [torch.as_tensor(x[1]) for x in model_predictions_list])
        gt = gt.to(predictions.device)
        if split_acc_based_on_labels:
            zero_acc, ones_acc = compute_label_wise_acc(gt, predictions)
            accuracy_dictionary[graph_size] = (zero_acc, ones_acc)
        else:
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
            skip_this_round = True
            return skip_this_round, None, None
    if is_plotting_enabled:
        plot_bar_plot(dictionary_to_plot=accuracy_dictionary, y_label='accuracy',
                      title=f'{model_type} accuracy vs. size',
                      filename=filename_acc, output_dir=output_dir)
        plot_bar_plot(dictionary_to_plot=roc_dictionary, y_label='roc', title=f'{model_type} roc vs. size',
                      filename=filename_roc, output_dir=output_dir, color='b')
    if output_dir is not None:
        cm_save_path = os.path.join(output_dir, f'cm{fold}.pkl')
        pickle.dump(cm_dict, open(cm_save_path, 'wb'))
    return skip_this_round, accuracy_dictionary, roc_dictionary


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


def eval_regr_loss(model, loader):
    # The computation  is defined only when we are working with regression target
    # Since mean operation is not supported on long tensors.
    return -1.0


def compute_label_wise_acc(gt, predictions):
    zero_indices = torch.where(gt == 0)[0]
    ones_indices = torch.where(gt == 1)[0]
    predicted_label = predictions.max(1)[1]
    zero_acc = predicted_label[zero_indices].eq(gt[zero_indices].view(-1)).sum().item() / zero_indices.shape[0]
    ones_acc = predicted_label[ones_indices].eq(gt[ones_indices].view(-1)).sum().item() / ones_indices.shape[0]
    return zero_acc, ones_acc


def display_mean_std(metric_name, metric_value_tensor):
    mean = metric_value_tensor.mean().item()
    std = metric_value_tensor.unsqueeze(0).std().item()
    print(f'Test {metric_name}: {mean:.3f} ± {std:.3f}')


def pretty_print_avg_dictionary(input_dict):
    for key, values in input_dict.items():
        print(f"{key}---------{sum(values) / len(values)}")


def print_custom_avg_of_dictionary(input_dict):
    """

    :param input_dict: A dictionary with string key and a list of values to reduce
    :param y_label: plot label
    :param filename: filename to save the plot
    :param output_dir: directory location for saving plots
    :param color: color of bar plot
    :return: None
    """
    # The input dictionary has
    # {"large": [acc_lab0, acc_lab1]}
    avg_dict = {}
    for key, nested_list in input_dict.items():
        list_zeros, list_ones = [], []
        for x in nested_list:
            list_zeros.append(x[0])
            list_ones.append(x[1])
        avg_dict[f"{key}_0"] = sum(list_zeros) / len(list_zeros)
        avg_dict[f"{key}_1"] = sum(list_ones) / len(list_ones)
    for key, value in avg_dict.items():
        print(f"{key} has the accuracy {value}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch.nn.functional as F


class FocalLoss(torch.nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='mean'):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction
        print(f"Using Focal loss with alpha {self.weight} and gamma {self.gamma}")

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


# Used from -> https://github.com/FLHonker/Losses-in-image-classification-task
class CenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=2.0):
        super(CenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label, batch_size=None):
        if batch_size is None:
            batch_size = feat.shape[0]
        expanded_centers = self.centers.index_select(dim=0, index=label)
        intra_distances = feat.dist(expanded_centers)
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances
        return loss


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super(ContrastiveCenterLoss, self).__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    def forward(self, feat, label):
        batch_size = feat.shape[0]

        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_feat = feat.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_feat - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, label.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
               (inter_distances + epsilon) / 0.1

        return loss


if __name__ == '__main__':
    x = torch.randn((2, 5))
    y = torch.as_tensor([0, 1])
    criterion = CenterLoss(feat_dim=5, num_classes=2, weights=torch.as_tensor([0.5, 1]))
    print(criterion(x, y))
    criterion = ContrastiveCenterLoss(feat_dim=5, num_classes=2)
    print(criterion(x, y))
