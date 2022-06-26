import argparse
import os
import time
import torch
from itertools import product
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from environment_setup import write_configs_to_disk, PROJECT_ROOT_DIR, \
    get_configurations_dtype_string
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
from graph_models.model_factory import get_model
from model_training.eval_utils import eval_loss, eval_roc_auc, eval_acc, eval_graph_len_acc, \
    plot_results_based_on_graph_size
from model_training.train_eval import train, k_fold
from utils.training_utils import LabelEncoder, LogWriterWrapper

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


layers = [2, 3]
hiddens = [32, 128, 256]


def final_cross_validation_with_val_set(dataset, model, folds, epochs, batch_size, lr, lr_decay_factor,
                                        lr_decay_step_size, weight_decay, logger, device, num_layers, hidden):
    val_losses, roc_auc, test_roc_auc, test_accuracy, durations = [], [], [], [], []
    for fold, (train_idx, test_idx,
               val_idx) in enumerate(zip(*k_fold(dataset, folds))):

        train_dataset = [dataset[idx.item()] for idx in train_idx]
        test_dataset = [dataset[idx.item()] for idx in test_idx]
        val_dataset = [dataset[idx.item()] for idx in val_idx]

        # We also need to obtain class weights to ensure we do not have data imbalance issues.
        pos_samples = sum([sample[1] for sample in train_dataset])
        neg_samples = len(train_dataset) - pos_samples
        if pos_samples > neg_samples:
            class_balance_weights = torch.as_tensor([pos_samples / neg_samples, 1], device=device)
        else:
            class_balance_weights = torch.as_tensor([1, neg_samples / pos_samples], device=device)
        # Creating the data loaders
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # Let us also create the One Hot Encoder needed for auroc computation
        enc = LabelEncoder()

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_balance_weights)
        # END: Model initialization

        # BEGIN: Logger creation
        base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                    get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
        log_dir = os.path.join(base_log_dir, f"_layers_{num_layers}_hidden_dim_{hidden}")
        os.makedirs(log_dir, exist_ok=True)
        train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'train_{fold}')))
        val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'val_{fold}')))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        train_val_loop(criterion, enc, epochs, logger, lr_decay_factor, lr_decay_step_size, model, optimizer, roc_auc,
                       train_loader, train_writer, val_loader, val_losses,
                       val_writer, log_dir, fold)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        # Now we will load the best model in the given fold
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(log_dir, f"model_{fold}.pth"))))
        test_roc_auc.append(eval_roc_auc(model, test_loader, enc, epoch=fold, writer=val_writer))
        test_accuracy.append(eval_acc(model, test_loader))
        durations.append(t_end - t_start)
        print(f"Test accuracy {test_accuracy[-1]} and Test ROC {test_roc_auc[-1]}")
        test_acc, size_cm_dict = eval_graph_len_acc(model, test_loader.dataset)
        assert test_acc == test_accuracy[-1], f"Value differs as {test_acc} vs. {test_accuracy[-1]}"
        plot_results_based_on_graph_size(size_cm_dict, output_dir=log_dir, filename_acc=f'acc_{fold}',
                                         filename_roc=f'roc_{fold}', model_type=model)

    loss, roc_auc, test_roc_auc, duration, test_accuracy = torch.as_tensor(val_losses), torch.as_tensor(
        roc_auc), torch.as_tensor(
        test_roc_auc), torch.as_tensor(durations), torch.as_tensor(test_accuracy)
    loss, roc_auc = loss.view(folds, epochs), roc_auc.view(folds, epochs)
    # Select the best model based on val

    loss_mean = loss.mean().item()
    test_roc_auc_mean = test_roc_auc.mean().item()
    test_roc_auc_std = test_roc_auc.unsqueeze(0).std().item()
    test_accuracy_mean = test_accuracy.mean().item()
    test_accuracy_std = test_accuracy.unsqueeze(0).std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test AUROC: {test_roc_auc_mean:.3f} '
          f'± {test_roc_auc_std:.3f}, Duration: {duration_mean:.3f}')

    return loss_mean, test_roc_auc_mean, test_roc_auc_std, test_accuracy_mean, test_accuracy_std


def train_val_loop(criterion, enc, epochs, logger, lr_decay_factor, lr_decay_step_size, model, optimizer, roc_auc,
                   train_loader, train_writer, val_loader, val_losses,
                   val_writer, log_dir, fold):
    best_val_roc = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader, criterion, epoch, train_writer)
        val_loss = eval_loss(model, val_loader, criterion, epoch, val_writer)
        val_roc = eval_roc_auc(model, val_loader, enc, epoch=epoch, writer=val_writer)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_roc': val_roc,
        }
        # Appending the results for selecting best model
        val_losses.append(val_loss)
        roc_auc.append(val_roc)

        if logger is not None:
            logger(eval_info)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            torch.save(model.state_dict(), os.path.join(log_dir, f"model_{fold}.pth"))


def main():
    seed_everything(seed=42)
    dataset = get_dataset()
    # Write configurations to the disk
    # Please do this from the main process.
    # Once we do it, the initial set of configs are persisted
    write_configs_to_disk()
    # Determines how many samples from random grid search are made
    num_folds = 5
    sample_graph_data = dataset[0][0]

    # Check if cuda available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    best_result = (float('inf'), 0, 0, 0, 0)  # (loss, auc, std, acc, acc_std)
    for num_layers, hidden in product(layers, hiddens):
        dataset = get_dataset()
        model = get_model(model_type='linear', hidden_dim=hidden, num_layers=num_layers,
                          sample_graph_data=sample_graph_data)
        loss, roc, std, acc, acc_std = final_cross_validation_with_val_set(
            dataset=dataset,
            model=model,
            folds=num_folds,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
            device=device,
            num_layers=num_layers,
            hidden=hidden
        )
        if roc > best_result[1]:
            best_result = (loss, roc, std, acc, acc_std)
            desc = f'{best_result[1]:.3f} ± {best_result[2]:.3f} and accuracy {best_result[3]:.3f} ± {best_result[4]:.3f}'
            print(f'{model}; Hidden:{hidden}, Layers:{num_layers}: {desc}')
            results += [f'{model}; Hidden:{hidden}, Layers:{num_layers}: {desc}']
    results = '\n'.join(results)
    print(f'--\n{results}')


if __name__ == '__main__':
    main()
