import argparse
import itertools
import os
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToSparseTensor

from dataset.dataset_factory import get_dataset
from environment_setup import write_configs_to_disk, PROJECT_ROOT_DIR, \
    get_configurations_dtype_string, get_configurations_dtype_string_list, device, get_configurations_dtype_boolean
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
from graph_models.model_factory import get_model
from utils.training_utils import eval_graph_len_acc, \
    LabelEncoder, LogWriterWrapper, get_dataset_and_auxiliary_loss, \
    get_class_weights, train_val_loop, k_fold, balanced_batch_sampler, seed_worker, g, \
    eval_roc_auc_with_prec_recall_acc_and_f1, display_mean_std

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=3 * 1e-4)
parser.add_argument('--check_ret_ratio', action='store_true')
args = parser.parse_args()


def final_cross_validation_with_val_set(dataset, model, folds, epochs, batch_size, lr, weight_decay, num_layers,
                                        hidden
                                        ):
    val_losses, roc_auc = [], []
    test_roc_auc_min_loss, test_accuracy_min_loss = [], []
    # Same for the extra metrics we can use in our evaluation
    prec_list_min_loss, recall_list_min_loss, acc_list_min_loss, f1_list_min_loss = [], [], [], []
    # For PCA visualization
    # Changing this piece to see the PCA
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    auxiliary_string = f"{get_configurations_dtype_boolean(section='TRAINING', key='USE_SIP')}"
    log_dir = os.path.join(base_log_dir + auxiliary_string, f"_layers_{num_layers}_hidden_dim_{hidden}")
    test_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, 'test')))
    out_feat = torch.FloatTensor().to(device)
    outGT = torch.FloatTensor().to(device)
    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(dataset, folds))):

        test_dataset, train_dataset, val_dataset = get_dataset_and_auxiliary_loss(dataset,
                                                                                  test_idx,
                                                                                  train_idx,
                                                                                  val_idx,
                                                                                  no_aug=True)

        class_balance_weights = get_class_weights(train_dataset)
        balanced_sampler = balanced_batch_sampler(train_dataset=train_dataset)
        # Creating the data loaders
        print(f"Class balance weight is {class_balance_weights}")
        train_loader = DataLoader(train_dataset, batch_size, sampler=balanced_sampler, worker_init_fn=seed_worker,
                                  generator=g)
        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        print(
            f"Starting training with train size:\nTrain: {len(train_dataset)}\nVal: {len(val_dataset)}\nTest: {len(test_dataset)}")

        # Let us also create the One Hot Encoder needed for auroc computation
        enc = LabelEncoder()

        model.to(device).reset_parameters()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = None
        criterion = nn.CrossEntropyLoss()

        # BEGIN: Logger creation
        os.makedirs(log_dir, exist_ok=True)
        train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'train_{fold}')))
        val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'val_{fold}')))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_val_loop(criterion=criterion, enc=enc, epochs=epochs, model=model, optimizer=optimizer,
                       roc_auc=roc_auc,
                       train_loader=train_loader, train_writer=train_writer, val_loader=val_loader,
                       val_losses=val_losses,
                       val_writer=val_writer, log_dir=log_dir, fold=fold,
                       scheduler=scheduler)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Now we will load the best model in the given fold
        model.reset_parameters()
        # Stats for the best model
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(log_dir, f"{model}_loss_{fold}.pth"))))
        roc_auc_min_loss, prec_min_loss, recall_min_loss, acc_min_loss, f1_min_loss = eval_roc_auc_with_prec_recall_acc_and_f1(
            model, test_loader, enc, epoch=fold, writer=None)
        test_roc_auc_min_loss.append(roc_auc_min_loss)
        prec_list_min_loss.append(prec_min_loss)
        recall_list_min_loss.append(recall_min_loss)
        acc_list_min_loss.append(acc_min_loss)
        f1_list_min_loss.append(f1_min_loss)

        test_acc, size_cm_dict = eval_graph_len_acc(model, test_dataset)
        test_accuracy_min_loss.append(test_acc)
        print(
            f"Test accuracy {test_acc}, Test ROC {test_roc_auc_min_loss[-1]}")

    # Accumulating the stats
    loss, roc_auc = torch.as_tensor(val_losses), torch.as_tensor(roc_auc)
    loss, roc_auc = loss.view(folds, epochs), roc_auc.view(folds, epochs)
    loss_mean = loss.mean().item()
    test_roc_auc_min_loss, test_accuracy_min_loss = torch.as_tensor(test_roc_auc_min_loss), torch.as_tensor(
        test_accuracy_min_loss)
    prec_tensor_min_loss, recall_tensor_min_loss, acc_tensor_min_loss, f1_tensor_min_loss = torch.as_tensor(
        prec_list_min_loss), torch.as_tensor(recall_list_min_loss), torch.as_tensor(acc_list_min_loss), torch.as_tensor(
        f1_list_min_loss)
    # loss, roc_auc = loss.view(folds, epochs), roc_auc.view(folds, epochs)
    test_roc_auc_min_loss_mean = test_roc_auc_min_loss.mean().item()
    test_roc_auc_min_loss_std = test_roc_auc_min_loss.unsqueeze(0).std().item()
    test_accuracy_min_loss_mean = test_accuracy_min_loss.mean().item()
    test_accuracy_min_loss_std = test_accuracy_min_loss.unsqueeze(0).std().item()
    print(
        f'Val Loss: {loss_mean:.4f}, Test AUROC: {test_roc_auc_min_loss_mean:.3f} ± {test_roc_auc_min_loss_std:.3f}')
    # The same for min_loss variant
    display_mean_std(metric_name='precision', metric_value_tensor=prec_tensor_min_loss)
    display_mean_std(metric_name='recall', metric_value_tensor=recall_tensor_min_loss)
    display_mean_std(metric_name='accuracy', metric_value_tensor=acc_tensor_min_loss)
    display_mean_std(metric_name='f1-score', metric_value_tensor=f1_tensor_min_loss)
    return loss_mean, test_roc_auc_min_loss_mean, test_roc_auc_min_loss_std, test_accuracy_min_loss_mean, \
           test_accuracy_min_loss_std


def main(model_type, retention_ratio):
    # Write configurations to the disk
    # Please do this from the main process.
    # Once we do it, the initial set of configs are persisted
    write_configs_to_disk()
    # Determines how many samples from random grid search are made
    num_folds = 10

    results = []
    num_layers = 2
    hidden = 64
    if model_type in ['gat']:
        print(f"{model_type} does not support edge_attr so dropping edge_attr information")
        dataset = get_dataset(transform=ToSparseTensor(attr=None))
    else:
        print("edge_attr supported. Including edge_attr information")
        dataset = get_dataset(transform=ToSparseTensor(attr='edge_attr'))
    print(dataset)
    sample_graph_data = dataset[0][0]
    print(sample_graph_data)
    model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                      sample_graph_data=sample_graph_data, retention_ratio=retention_ratio)
    print(model.get_full_des())
    loss, roc_mil, roc_std_mil, acc_mil, acc_std_mil = final_cross_validation_with_val_set(
        dataset=dataset,
        model=model,
        folds=num_folds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-2,
        num_layers=num_layers,
        hidden=hidden,
    )
    desc = f'{roc_mil:.3f} ± {roc_std_mil:.3f}, accuracy {acc_mil:.3f} ± {acc_std_mil:.3f}'
    results += [f'{model}: {desc}']
    print(f'--\n{results}')
    return roc_mil


if __name__ == '__main__':
    model_types = get_configurations_dtype_string_list(section='TRAINING', key='MODEL_TYPES')
    # Configuration to enable check for different retention ratios.
    # retention_ratio = np.arange(0.1, 1, 0.1) if args.check_ret_ratio else [0.1]
    retention_ratio = [0.3, 0.4, 0.5, 0.6, 0.7] if args.check_ret_ratio else [0.5]
    model_roc_dict, model_loss_dict = defaultdict(list), defaultdict(list)
    for model_type, ret_ratio in itertools.product(model_types, retention_ratio):
        seed_everything(seed=42)
        cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        print(f"Using pair {model_type} and ret-ratio {ret_ratio}")
        roc_mil = main(model_type=model_type, retention_ratio=ret_ratio)
        model_loss_dict[f'{model_type}_{ret_ratio}'].append(roc_mil)

    print("Best loss variant")
    for key, value in model_loss_dict.items():
        print(f"{key} has {value} with mean of {np.mean(value)}")
