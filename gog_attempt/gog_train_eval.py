import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.backends import cudnn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything

from torch_geometric.utils import index_to_mask
from tqdm import tqdm

from environment_setup import write_configs_to_disk, PROJECT_ROOT_DIR, \
    get_configurations_dtype_string
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
from gog_attempt.GoGDataset import GoGDataset
from gog_attempt.GoGModel import GoGModel
from model_training.train_eval import k_fold
from utils.training_utils import LabelEncoder, LogWriterWrapper, shuffle_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=150)
parser.add_argument('--lr', type=float, default=3 * 1e-4)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
parser.add_argument('--batch_size', type=int, default=16)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


def gog_cross_validation(dataset, model, folds, epochs, batch_size, lr, lr_decay_factor,
                         lr_decay_step_size, weight_decay, logger, device, num_layers, hidden
                         ):
    val_losses, roc_auc, test_roc_auc, test_accuracy, test_regr_loss, durations = [], [], [], [], [], []
    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(dataset, folds))):
        train_mask, val_mask, test_mask = index_to_mask(train_idx, size=len(dataset)), index_to_mask(val_idx, size=len(
            dataset)), index_to_mask(test_idx, size=len(dataset))
        criterion_vol_regr = nn.L1Loss()
        # We also need to obtain class weights to ensure we do not have data imbalance issues.
        # Since we have used stratified sampling, we should get near identical values for the weights.
        pos_samples = sum([sample[1] for sample in dataset])
        neg_samples = len(dataset) - pos_samples
        if pos_samples > neg_samples:
            class_balance_weights = torch.as_tensor([pos_samples / neg_samples, 1], device=device)
        else:
            class_balance_weights = torch.as_tensor([1, neg_samples / pos_samples], device=device)
        # Creating the data loaders
        print(f"Class balance weight is {class_balance_weights}")
        # Let us also create the One Hot Encoder needed for auroc computation
        enc = LabelEncoder()

        model.to(device).reset_parameters()
        # We distinguish between "individual_graph_proc" and "GoGModel" parameters.
        optimizer = Adam([
            {'params': model.individual_graph_proc.parameters(), 'lr': args.lr/100},
            {'params': model.node_processor.parameters()},
            {'params': model.lin1.parameters()},
            {'params': model.lin2.parameters()},
        ], lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss(weight=class_balance_weights)

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

        train_val_loop(criterion=criterion, enc=enc, epochs=epochs, logger=logger, lr_decay_factor=lr_decay_factor,
                       lr_decay_step_size=lr_decay_step_size, model=model, optimizer=optimizer,
                       train_writer=train_writer, dataset=dataset, val_writer=val_writer, log_dir=log_dir,
                       fold=fold, criterion_vol_regr=criterion_vol_regr, train_mask=train_mask, val_mask=val_mask)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        # Now we will load the best model in the given fold
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(log_dir, f"{model}_{fold}.pth"))))
        test_info = evaluate_gog(model=model, criterion=criterion, dataset=dataset, enc=enc,
                                 criterion_vol_regr=criterion_vol_regr, mask=test_mask)
        test_roc_auc.append(test_info['roc'])
        test_accuracy.append(test_info['acc'])
        test_regr_loss.append(test_info['loss'])
        durations.append(t_end - t_start)
        print(f"Test accuracy {test_accuracy[-1]}, Test ROC {test_roc_auc[-1]} and Test Regr loss {test_regr_loss[-1]}")

    loss, roc_auc, test_roc_auc, duration, test_accuracy, test_regr = torch.as_tensor(val_losses), torch.as_tensor(
        roc_auc), torch.as_tensor(
        test_roc_auc), torch.as_tensor(durations), torch.as_tensor(test_accuracy), torch.as_tensor(test_regr_loss)

    loss_mean = loss.mean().item()
    test_roc_auc_mean = test_roc_auc.mean().item()
    test_roc_auc_std = test_roc_auc.unsqueeze(0).std().item()
    test_accuracy_mean = test_accuracy.mean().item()
    test_accuracy_std = test_accuracy.unsqueeze(0).std().item()
    test_regr_mean = test_regr.mean().item()
    test_regr_std = test_regr.unsqueeze(0).std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test AUROC: {test_roc_auc_mean:.3f} '
          f'± {test_roc_auc_std:.3f},'
          f' Test Regr: {test_regr_mean} ± {test_regr_std:.3f},'
          f' Duration: {duration_mean:.3f}')

    return loss_mean, test_roc_auc_mean, test_roc_auc_std, test_accuracy_mean, test_accuracy_std, test_regr_mean, test_regr_std


def evaluate_gog(model, criterion, dataset, enc, criterion_vol_regr, mask):
    model.eval()
    correct, loss = 0, 0
    with torch.no_grad():
        logits, labels = model(dataset)
        loss += criterion(logits[mask], labels[mask]).item()
        pred = logits[mask].max(1)[1]
        correct += pred.eq(labels[mask]).sum().item()
        outPRED = logits[mask]
        outGT = labels[mask]

    predictions = torch.softmax(outPRED, dim=1)
    predictions, target = predictions.cpu().numpy(), outGT.cpu().numpy()
    # Encoder is callable.
    # Hence, we execute callable which returns the self.encoder instance
    target_one_hot = enc().transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    accuracy = correct / mask.sum()
    eval_info = {
        "loss": loss,
        "acc": accuracy,
        "roc": roc_auc_value
    }
    return eval_info


def train_gog(model, optimizer, criterion, dataset, criterion_vol_regr, train_mask):
    model.train()
    logits, labels = model(dataset)
    # Select the labels that belong to the training set only
    optimizer.zero_grad()
    loss = criterion(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()
    return loss


def train_val_loop(criterion, enc, epochs, logger, lr_decay_factor, lr_decay_step_size, model, optimizer,
                   dataset, train_writer, val_writer, log_dir, fold, train_mask, val_mask, criterion_vol_regr
                   ):
    best_val_roc = 0
    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = train_gog(model=model, optimizer=optimizer, criterion=criterion, dataset=dataset,
                  criterion_vol_regr=criterion_vol_regr, train_mask=train_mask)
        eval_info = evaluate_gog(model=model, criterion=criterion, dataset=dataset, enc=enc,
                                 criterion_vol_regr=criterion_vol_regr, mask=val_mask)
        train_info = evaluate_gog(model=model, criterion=criterion, dataset=dataset, enc=enc,
                                  criterion_vol_regr=criterion_vol_regr, mask=train_mask)
        eval_info['epoch'] = epoch

        if logger is not None:
            logger(eval_info)

        if eval_info['roc'] > best_val_roc:
            best_val_roc = eval_info['roc']
            torch.save(model.state_dict(), os.path.join(log_dir, f"{model}_{fold}.pth"))
        train_writer.add_scalar('roc', train_info['roc'], global_step=epoch)
        val_writer.add_scalar('roc', eval_info['roc'], global_step=epoch)
        train_writer.add_scalar('loss', train_loss, global_step=epoch)
        val_writer.add_scalar('loss', eval_info['loss'], global_step=epoch)
        train_writer.add_scalar('accuracy', train_info['acc'], global_step=epoch)
        val_writer.add_scalar('accuracy', eval_info['acc'], global_step=epoch)


@torch.no_grad()
def inference(model, data):
    model.eval()
    model(data)


def main():
    # Write configurations to the disk
    # Please do this from the main process.
    # Once we do it, the initial set of configs are persisted
    write_configs_to_disk()
    num_folds = 10
    results = []
    num_layers = 2
    hidden = 64
    dataset = GoGDataset()[0]
    print(dataset)
    model = GoGModel(hidden_dim=hidden, node_feature_dim=772, num_classes=2, total_number_of_gnn_layers=num_layers)
    print(model.get_full_des())
    loss, roc, roc_std, acc, acc_std, regr, regr_std = gog_cross_validation(
        dataset=dataset,
        model=model,
        folds=num_folds,
        epochs=args.epochs,
        lr=args.lr,
        lr_decay_factor=args.lr_decay_factor,
        lr_decay_step_size=args.lr_decay_step_size,
        weight_decay=1e-4,
        logger=None,
        device=device,
        num_layers=num_layers,
        hidden=hidden,
        batch_size=args.batch_size
    )
    desc = f'{roc:.3f} ± {roc_std:.3f}, accuracy {acc:.3f} ± {acc_std:.3f} and regr {regr:.3f} ± {regr_std:.3f}'
    results += [f'{model}: {desc}']
    print(f'--\n{results}')
    return loss, roc, roc_std, acc, acc_std, regr, regr_std


if __name__ == '__main__':
    seed_everything(seed=42)
    cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    best_result_iter = []
    for _ in range(5):
        best_result_iter.append(np.asarray(main()))
    # Now we can take the mean value for the different executions
    # and report the final result
    best_result = np.asarray(best_result_iter)
    print(f"For Gog model, result is {np.mean(best_result, axis=0)}")
