from collections import defaultdict

import argparse
import numpy as np
import os
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from environment_setup import write_configs_to_disk, PROJECT_ROOT_DIR, \
    get_configurations_dtype_string
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
from graph_models.model_factory import get_model
from model_training.eval_utils import eval_roc_auc, eval_acc, eval_graph_len_acc, \
    plot_results_based_on_graph_size, eval_acc_with_confusion_matrix, plot_avg_of_dictionary
from model_training.train_eval import k_fold
from utils.training_utils import LabelEncoder

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataset.dataset_factory import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=400)
args = parser.parse_args()


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


def stats_for_each_split(dataset, model, folds, batch_size,
                         logger, device, checkpoint_dir):
    test_roc_auc, test_accuracy, test_cm_list = [], [], []
    acc_dict_avg, roc_dict_avg = defaultdict(list), defaultdict(list)
    for fold, (_, _, test_idx) in enumerate(zip(*k_fold(dataset, folds))):

        test_dataset = [dataset[idx.item()] for idx in test_idx]
        # We also need to obtain class weights to ensure we do not have data imbalance issues.
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        # Let us also create the One Hot Encoder needed for auroc computation
        enc = LabelEncoder()
        model.to(device).reset_parameters()
        base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                    get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        test_images_dir = os.path.join(base_log_dir, 'final_test_images')
        os.makedirs(test_images_dir, exist_ok=True)
        # We will load the best model in the given fold.
        # The checkpoints are already saved. We would simply load the values
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_{fold}.pth"))))
        test_roc_auc.append(eval_roc_auc(model, test_loader, enc, epoch=fold, writer=None))
        acc, cm = eval_acc_with_confusion_matrix(model, test_loader)
        # Append both sets of results
        test_accuracy.append(acc)
        test_cm_list.append(cm)
        print(f"Test accuracy {test_accuracy[-1]} and Test ROC {test_roc_auc[-1]}")
        test_acc, size_cm_dict = eval_graph_len_acc(model, test_loader.dataset)
        assert test_acc == test_accuracy[-1], f"Value differs as {test_acc} vs. {test_accuracy[-1]}"
        acc_dict, roc_dict = plot_results_based_on_graph_size(size_cm_dict, output_dir=test_images_dir, filename_acc=f'acc_{fold}',
                                         filename_roc=f'roc_{fold}', model_type=model, fold=fold, is_plotting_enabled=False)
        # Let us add the values to our final dictionary so that we can average it
        for key, _ in acc_dict.items():
            acc_dict_avg[key].append(acc_dict[key])
            roc_dict_avg[key].append(roc_dict[key])
    test_roc_auc, test_accuracy = torch.as_tensor(test_roc_auc), torch.as_tensor(test_accuracy)

    test_roc_auc_mean = test_roc_auc.mean().item()
    test_roc_auc_std = test_roc_auc.unsqueeze(0).std().item()
    test_accuracy_mean = test_accuracy.mean().item()
    test_accuracy_std = test_accuracy.unsqueeze(0).std().item()
    print(f'Test AUROC: {test_roc_auc_mean:.3f} '
          f'± {test_roc_auc_std:.3f}')
    print(f"The final confusion matrix across {folds} is:-")
    final_cm = np.sum(test_cm_list, axis=0)
    print(final_cm)
    # Let us plot the final dictionary
    plot_avg_of_dictionary(input_dict=roc_dict_avg, y_label='roc', filename=f'{model}_roc_avg_all_folds', output_dir=test_images_dir, color='r')
    plot_avg_of_dictionary(input_dict=acc_dict_avg, y_label='acc', filename=f'{model}_acc_avg_all_folds', output_dir=test_images_dir, color='r')
    return test_roc_auc_mean, test_roc_auc_std, test_accuracy_mean, test_accuracy_std


def main(checkpoint_dir, model_type, hidden, num_layers):
    seed_everything(seed=42)
    dataset = get_dataset()
    # Write configurations to the disk
    # Please do this from the main process.
    # Once we do it, the initial set of configs are persisted
    write_configs_to_disk()
    # Determines how many samples from random grid search are made
    num_folds = 10
    sample_graph_data = dataset[0][0]

    # Check if cuda available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset()
    model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                      sample_graph_data=sample_graph_data)
    roc, std, acc, acc_std = stats_for_each_split(
        dataset=dataset,
        model=model,
        folds=num_folds,
        batch_size=args.batch_size,
        logger=None,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )
    desc = f'{roc:.3f} ± {std:.3f} and accuracy {acc:.3f} ± {acc_std:.3f}'
    print(f'{model}; Hidden:{hidden}, Layers:{num_layers}: {desc}')


if __name__ == '__main__':
    hidden = 256
    num_layers = 2
    model_type = 'gat'
    checkpoint_dir = os.path.join(PROJECT_ROOT_DIR,
                                  get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'),
                                  f'_layers_{num_layers}_hidden_dim_{hidden}'
                                  )

    main(checkpoint_dir=checkpoint_dir, model_type=model_type, hidden=hidden, num_layers=num_layers)
