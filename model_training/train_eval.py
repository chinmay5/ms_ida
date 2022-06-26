import numpy as np
import os
import pickle
import torch
from functools import partial
# ray tune for hyper-parameter optimization
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import StratifiedKFold
from torch import tensor, nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR
from graph_models.model_factory import get_model
from model_training.eval_utils import eval_loss, eval_acc, eval_roc_auc
from utils.training_utils import LogWriterWrapper, LabelEncoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def cross_validation_with_val_set(dataset, folds, model_type, epochs, batch_size,
                                  lr_decay_factor, lr_decay_step_size,
                                  weight_decay, num_samples, sample_graph_data, logger=None):
    test_rocs, max_val_rocs = [], []
    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*k_fold(dataset, folds))):

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
        print(class_balance_weights)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

        # Let us also create the One Hot Encoder needed for auroc computation
        enc = LabelEncoder()
        # Parameters specific to ray tune
        config = {
            "hidden": tune.sample_from(lambda _: 2 ** np.random.randint(4, 9)),
            "num_layers": tune.grid_search([2, 3, 4]),
            "lr": tune.loguniform(1e-5, 1e-1),
        }

        scheduler = ASHAScheduler(
            mode="max",
            max_t=epochs,
            grace_period=1,
            reduction_factor=2)
        reporter = CLIReporter(
            metric_columns=["roc_auc", "hidden", "num_layers", "lr"])

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        result = tune.run(
            partial(train_and_save_best_model,
                    weight_decay=weight_decay, class_balance_weights=class_balance_weights, fold=fold,
                    epochs=epochs, logger=logger, lr_decay_factor=lr_decay_factor,
                    lr_decay_step_size=lr_decay_step_size, train_loader=train_loader, val_loader=val_loader,
                    sample_graph_data=sample_graph_data, enc=enc, model_type=model_type),
            resources_per_trial={"cpu": 8, "gpu": 1},
            config=config,
            num_samples=num_samples,
            scheduler=scheduler,
            progress_reporter=reporter,
            local_dir=get_configurations_dtype_string(section='TRAINING', key='LOG_DIR')
        )

        best_trial = result.get_best_trial("roc_auc", "max", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["roc_auc"]))
        # Using the best model
        best_trained_model = get_model(model_type=model_type, hidden_dim=best_trial.config["hidden"],
                                       num_layers=best_trial.config["num_layers"], sample_graph_data=sample_graph_data)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state = torch.load(os.path.join(
            best_checkpoint_dir, f"checkpoint_{fold}.pth"))
        best_trained_model.load_state_dict(model_state)

        test_roc = eval_roc_auc(best_trained_model, test_loader, enc)
        test_rocs.append(test_roc)
        print("Best trial test set accuracy: {}".format(test_roc))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Although we are looking at all the test values, it is sort of a sanity.
    # We are still using the accuracy corresponding to the best performing validation loss.
    test_roc = tensor(test_rocs)
    max_val_roc = tensor(max_val_rocs)
    return test_roc.mean().item(), torch.std(test_roc).item()


def train_and_save_best_model(config, weight_decay, class_balance_weights, fold, epochs, logger, lr_decay_factor,
                              lr_decay_step_size,
                              train_loader, val_loader, sample_graph_data, enc, model_type):
    max_val_roc_auc = 0
    # BEGIN: Model Initialization
    model = get_model(model_type=model_type, hidden_dim=config["hidden"], num_layers=config["num_layers"],
                      sample_graph_data=sample_graph_data)
    print(model)
    model.to(device).reset_parameters()
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_balance_weights)
    # END: Model initialization

    # BEGIN: Logger creation
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    log_dir = os.path.join(base_log_dir, f"_layers_{config['num_layers']}_hidden_dim_{config['hidden']}")
    os.makedirs(log_dir, exist_ok=True)
    train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'train_{fold}')))
    val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'val_{fold}')))
    # END: Logger creation

    # BEGIN: Model training
    for epoch in range(1, epochs + 1):
        train_loss = train(model, optimizer, train_loader, criterion, epoch, train_writer)
        val_loss = eval_loss(model, val_loader, criterion, epoch, val_writer)
        val_acc = eval_acc(model, val_loader)
        val_roc = eval_roc_auc(model, val_loader, enc, epoch=epoch, writer=val_writer)
        eval_info = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_roc': val_roc,
            'val_acc': val_acc,
        }

        if logger is not None:
            logger(eval_info)

        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_decay_factor * param_group['lr']

        if val_roc > max_val_roc_auc:
            max_val_roc_auc = val_roc
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, f"checkpoint_{fold}.pth")
                torch.save(model.state_dict(), path)
        # Adding info to raytune
        tune.report(roc_auc=val_roc)


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
        for _, idx in skf.split(torch.zeros(len(dataset)), dataset.graph_catogory_label.cpu().numpy().tolist()):
            test_indices.append(torch.from_numpy(idx).to(torch.long))

        val_indices = [test_indices[i - 1] for i in range(folds)]

        for i in range(folds):
            train_mask = torch.ones(len(dataset), dtype=torch.bool)
            train_mask[test_indices[i]] = 0
            train_mask[val_indices[i]] = 0
            train_indices.append(train_mask.nonzero(as_tuple=False).view(-1))
        # Now, let us go ahead and save these values
        os.makedirs(k_fold_split_path, exist_ok=False)  # Exists ok is not fine here.
        pickle.dump(train_indices, open(os.path.join(k_fold_split_path, "train_indices.pkl"), 'wb'))
        pickle.dump(val_indices, open(os.path.join(k_fold_split_path, "val_indices.pkl"), 'wb'))
        pickle.dump(test_indices, open(os.path.join(k_fold_split_path, "test_indices.pkl"), 'wb'))
        pickle.dump(folds, open(os.path.join(k_fold_split_path, "num_splits.pkl"), 'wb'))

    return train_indices, val_indices, test_indices


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
