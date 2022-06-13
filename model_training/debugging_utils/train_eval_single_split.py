from functools import partial

import numpy as np
import os
import pickle
import time
import torch
from itertools import product
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from dataset.dataset_factory import get_dataset
from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR, get_configurations_dtype_boolean
from graph_models.model_factory import get_model
from utils.training_utils import LogWriterWrapper, LabelEncoder

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_val_model(config, sample_graph_data, epochs, batch_size,
                    lr_decay_factor, lr_decay_step_size,
                    weight_decay, logger=None, base_log_dir=None):
    # We would directly load the datasets here.
    model = get_model(model_type='gcn', hidden_dim=config["hidden"], num_layers=config["num_layers"], sample_graph_data=sample_graph_data)
    print(model)
    log_dir = os.path.join(base_log_dir, f"_layers_{config['num_layers']}_hidden_dim_{config['hidden']}")
    os.makedirs(log_dir, exist_ok=True)
    temp_folder_path = get_configurations_dtype_string(section='SETUP', key='TEMP_FOLDER_PATH')
    if get_configurations_dtype_boolean(section='TRAINING', key='IS_HETERO'):
        train_dataset = pickle.load(open(os.path.join(temp_folder_path, 'train_set_het.pkl'), 'rb'))
        val_dataset = pickle.load(open(os.path.join(temp_folder_path, 'val_set_het.pkl'), 'rb'))
    else:
        train_dataset = pickle.load(open(os.path.join(temp_folder_path, 'train_set_hom.pkl'), 'rb'))
        val_dataset = pickle.load(open(os.path.join(temp_folder_path, 'val_set_hom.pkl'), 'rb'))
    # We also need to obtain class weights to ensure we do not have data imbalance issues.
    pos_samples = sum([sample[1] for sample in train_dataset])
    neg_samples = len(train_dataset) - pos_samples
    if pos_samples > neg_samples:
        class_balance_weights = torch.as_tensor([pos_samples / neg_samples, 1], device=device)
    else:
        class_balance_weights = torch.as_tensor([1, neg_samples / pos_samples], device=device)
    print(class_balance_weights)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)


    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_balance_weights)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t_start = time.perf_counter()
    # Let us create some tensorboard logging writers
    train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'train_single_split')))
    val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'val_single_split')))

    # Let us also create the One Hot Encoder needed for auroc computation
    enc = LabelEncoder()
    best_val_rocs = 0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader, criterion, epoch, train_writer)
        val_loss = eval_loss(model, val_loader, criterion, epoch, val_writer)
        val_acc = eval_acc(model, val_loader)
        val_rocs = eval_roc_auc(model, val_loader, enc, epoch=epoch, writer=val_writer)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_roc': val_rocs,
        }

        if logger is not None:
            logger(eval_info)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if val_rocs > best_val_rocs:
            best_val_rocs = val_rocs
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save(model.state_dict(), path)

        tune.report(roc_auc=val_rocs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    # Let us load the best model
    # model.load_state_dict(torch.load('best_model.pth'))



    t_end = time.perf_counter()

    # print(durations)
    # return test_roc


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, criterion, epoch, train_writer):
    model.train()

    # Some information needed for logging on tensorboard
    total_loss = 0
    for idx, (data, label) in enumerate(loader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        out = model(data)
        loss = criterion(out, label.view(-1))
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    avg_train_loss = total_loss / len(loader)
    train_writer.add_scalar('loss', avg_train_loss, global_step=epoch)
    return avg_train_loss


def eval_acc(model, loader):
    model.eval()
    correct = 0
    for data, labels in loader:
        data, labels = data.to(device), labels.to(device)
        with torch.no_grad():
            pred = model(data).max(1)[1]
        correct += pred.eq(labels.view(-1)).sum().item()
    return correct / len(loader.dataset)


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
    logging_base = epoch * len(loader)
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


def main():
    seed_everything(seed=42)
    dataset = get_dataset()
    sample_graph_data = dataset[0][0]
    # BEGIN: Hyper-parameters
    layers, hiddens = [2], [32, 128]
    epochs = 300
    batch_size = 128
    lr_decay_factor = 0.5
    lr_decay_step_size = 400
    lr = 1e-4
    num_samples=10
    # END: Hyperparameters
    config = {
        "hidden": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "num_layers": tune.grid_search([2, 3]),
        "lr": tune.loguniform(1e-5, 1e-1),
    }

    base_log_dir = os.path.join(PROJECT_ROOT_DIR, get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))

    scheduler = ASHAScheduler(
        mode="max",
        max_t=epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["roc_auc", "hidden", "num_layers", "lr"])


    result = tune.run(
        partial(train_val_model,
            sample_graph_data=sample_graph_data,
            epochs=epochs,
            batch_size=batch_size,
            lr_decay_factor=lr_decay_factor,
            lr_decay_step_size=lr_decay_step_size,
            weight_decay=1e-5,
            logger=None,
            base_log_dir=base_log_dir),
        resources_per_trial={"cpu": 8, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("roc_auc", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["roc_auc"]))

    temp_folder_path = get_configurations_dtype_string(section='SETUP', key='TEMP_FOLDER_PATH')
    test_dataset = pickle.load(open(os.path.join(temp_folder_path, 'test_set.pkl'), 'rb'))
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)


    best_trained_model = get_model(hidden_dim=best_trial.config["hidden"],
                                   num_layers=best_trial.config["num_layers"], sample_graph_data=sample_graph_data)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    enc = LabelEncoder()
    test_roc = eval_roc_auc(best_trained_model, test_loader, enc)
    print("Best trial test set accuracy: {}".format(test_roc))

    # for num_layers, hidden in product(layers, hiddens):
    #     model = get_model(hidden_dim=config["hidden"], num_layers=num_layers, sample_graph_data=sample_graph_data)
    #     log_dir = os.path.join(base_log_dir, f"_layers_{num_layers}_hidden_dim_{hidden}")
    #     os.makedirs(log_dir, exist_ok=True)
    #     test_roc = train_val_model(
    #         model=model,
    #         epochs=epochs,
    #         batch_size=batch_size,
    #         lr=lr,
    #         lr_decay_factor=lr_decay_factor,
    #         lr_decay_step_size=lr_decay_step_size,
    #         weight_decay=1e-5,
    #         logger=None,
    #         log_dir=log_dir,
    #     )
    #     print(f"For layers {num_layers} and hidden size {hidden}, the test ROC is {test_roc}")


if __name__ == '__main__':
    main()