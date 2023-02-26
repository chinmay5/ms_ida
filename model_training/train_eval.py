import os
import pickle
import torch
from functools import partial
# ray tune for hyper-parameter optimization
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader

from environment_setup import get_configurations_dtype_string, PROJECT_ROOT_DIR, device
from graph_models.model_factory import get_model
from utils.training_utils import LogWriterWrapper, LabelEncoder, get_dataset_and_auxiliary_loss, get_class_weights, \
    get_training_enhancements, k_fold, train_val_loop, eval_acc, eval_roc_auc, eval_regr_loss


def cross_validation_with_val_set(dataset, folds, model_type, epochs, batch_size,
                                  lr_decay_factor, lr_decay_step_size,
                                  weight_decay, num_samples, sample_graph_data, logger=None):
    val_losses, roc_auc, test_regr_loss_min_loss = [], [], [],
    test_roc_auc, test_regr_loss, test_accuracy = [], [], []

    for fold, (train_idx, val_idx, test_idx) in enumerate(zip(*k_fold(dataset, folds))):

        criterion_vol_regr, test_dataset, train_dataset, val_dataset = get_dataset_and_auxiliary_loss(dataset,
                                                                                                      test_idx,
                                                                                                      train_idx,
                                                                                                      val_idx)

        class_balance_weights = get_class_weights(train_dataset)
        print(f"Class balance weight is {class_balance_weights}")
        # train_loader = DataLoader(train_dataset, batch_size, sampler=graph_length_based_weights_sampler)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, worker_init_fn=seed_worker, generator=g)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        print(
            f"Starting training with train size:\nTrain: {len(train_dataset)}\nVal: {len(val_dataset)}\nTest: {len(test_dataset)}")

        # Let us also create the One Hot Encoder needed for auroc computation
        enc = LabelEncoder()
        # Parameters specific to ray tune
        config = {
            "hidden": tune.grid_search([64, 128, 256]),
            "num_layers": tune.grid_search([2, 3]),
            "lr": tune.loguniform(1e-5, 1e-3),
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
                    sample_graph_data=sample_graph_data, enc=enc, model_type=model_type,
                    criterion_vol_regr=criterion_vol_regr, train_dataset=train_dataset, batch_size=batch_size,
                    dataset=dataset, train_idx=train_idx, scheduler=None, val_losses=val_losses, roc_auc=roc_auc,
                    ),
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

        test_roc_auc.append(
            eval_roc_auc(best_trained_model, test_loader, enc, epoch=fold, writer=None,
                         criterion_vol_regr=criterion_vol_regr))
        test_regr_loss.append(eval_regr_loss(best_trained_model, test_loader, criterion_vol_regr=criterion_vol_regr))

        test_accuracy.append(
            eval_acc(best_trained_model, test_loader, criterion_vol_regr=criterion_vol_regr, writer=None))

        print(f"Test accuracy {test_accuracy[-1]}, Test ROC {test_roc_auc[-1]} and Test Regr loss {test_regr_loss[-1]}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Although we are looking at all the test values, it is sort of a sanity.
    # We are still using the accuracy corresponding to the best performing validation loss.

    loss, roc_auc, test_roc_auc, test_accuracy, test_regr = torch.as_tensor(val_losses), torch.as_tensor(
        roc_auc), torch.as_tensor(test_roc_auc), torch.as_tensor(test_accuracy), torch.as_tensor(test_regr_loss)

    loss, roc_auc = loss.view(folds, epochs), roc_auc.view(folds, epochs)
    # Select the best model based on val

    loss_mean = loss.mean().item()
    test_roc_auc_mean = test_roc_auc.mean().item()
    test_roc_auc_std = test_roc_auc.unsqueeze(0).std().item()
    test_accuracy_mean = test_accuracy.mean().item()
    test_accuracy_std = test_accuracy.unsqueeze(0).std().item()
    test_regr_mean = test_regr.mean().item()
    test_regr_std = test_regr.unsqueeze(0).std().item()

    return loss_mean, test_roc_auc_mean, test_roc_auc_std, test_accuracy_mean, test_accuracy_std, test_regr_mean, \
           test_regr_std


def train_and_save_best_model(config, weight_decay, class_balance_weights, fold, epochs, logger, lr_decay_factor,
                              lr_decay_step_size,
                              train_loader, val_loader, sample_graph_data, enc, model_type, criterion_vol_regr,
                              train_dataset, batch_size, dataset, train_idx, scheduler, val_losses, roc_auc):
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

    # BEGIN: Get training enhancements
    contras_trainer, mixup_trainer, mixup_train_loader, feature_alignment_loss, optimizer_centre_loss, train_loader = get_training_enhancements(
        criterion=criterion,
        criterion_vol_regr=criterion_vol_regr,
        model=model, optimizer=optimizer,
        train_writer=train_writer, train_dataset=train_dataset, batch_size=batch_size, lr=config["lr"],
        weight_decay=weight_decay, class_balance_weights=class_balance_weights, train_loader=train_loader)
    # END: Get training enhancements

    # BEGIN: Model training
    train_val_loop(criterion=criterion, enc=enc, epochs=epochs, logger=logger, lr_decay_factor=lr_decay_factor,
                   lr_decay_step_size=lr_decay_step_size, model=model, optimizer=optimizer, roc_auc=roc_auc,
                   train_loader=train_loader, train_writer=train_writer, val_loader=val_loader,
                   val_losses=val_losses,
                   val_writer=val_writer, log_dir=log_dir, fold=fold, criterion_vol_regr=criterion_vol_regr,
                   dataset_refresh_metadata=None, #(dataset, train_idx),
                   mixup_trainer=mixup_trainer, mixup_train_loader=mixup_train_loader,
                   contras_trainer=contras_trainer,
                   feature_alignment_loss_and_optim=[feature_alignment_loss, optimizer_centre_loss],
                   scheduler=scheduler, tune_obj=tune)


def num_graphs(data):
    if hasattr(data, 'num_graphs'):
        return data.num_graphs
    else:
        return data.x.size(0)

