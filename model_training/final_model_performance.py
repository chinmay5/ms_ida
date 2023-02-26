import argparse
import os
import time
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
from utils.training_utils import eval_roc_auc, eval_regr_loss, eval_graph_len_acc, \
    plot_results_based_on_graph_size, LabelEncoder, LogWriterWrapper, get_dataset_and_auxiliary_loss, \
    get_class_weights, get_training_enhancements, train_val_loop, k_fold, balanced_batch_sampler, seed_worker, g

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--lr', type=float, default=3 * 1e-4)
args = parser.parse_args()


def final_cross_validation_with_val_set(dataset, model, folds, epochs, batch_size, lr, weight_decay, num_layers,
                                        hidden
                                        ):
    val_losses, roc_auc, test_roc_auc, test_accuracy, test_regr_loss, durations = [], [], [], [], [], []
    test_roc_auc_min_loss, test_accuracy_min_loss, test_regr_loss_min_loss = [], [], []
    # For PCA visualization
    # Changing this piece to see the PCA
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    auxiliary_string = f"mixup_{get_configurations_dtype_boolean(section='TRAINING', key='USE_MIXUP')}_sip_" \
                       f"{get_configurations_dtype_boolean(section='TRAINING', key='USE_SIP')}"
    log_dir = os.path.join(base_log_dir + auxiliary_string, f"_layers_{num_layers}_hidden_dim_{hidden}")
    test_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, 'test')))
    out_feat = torch.FloatTensor().to(device)
    outGT = torch.FloatTensor().to(device)
    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(dataset, folds))):

        criterion_vol_regr, test_dataset, train_dataset, val_dataset = get_dataset_and_auxiliary_loss(dataset,
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
        scheduler = None  # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, min_lr=1e-7, patience=30)

        # criterion = nn.CrossEntropyLoss(weight=class_balance_weights)
        criterion = nn.CrossEntropyLoss()

        # BEGIN: Logger creation
        os.makedirs(log_dir, exist_ok=True)
        train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'train_{fold}')))
        val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'val_{fold}')))

        contras_trainer, mixup_trainer, mixup_train_loader, feature_alignment_loss, optimizer_centre_loss, train_loader = get_training_enhancements(
            criterion=criterion,
            criterion_vol_regr=criterion_vol_regr,
            model=model, optimizer=optimizer,
            train_writer=train_writer, train_dataset=train_dataset, batch_size=batch_size, lr=lr,
            weight_decay=weight_decay, class_balance_weights=class_balance_weights, train_loader=train_loader)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        train_val_loop(criterion=criterion, enc=enc, epochs=epochs, model=model, optimizer=optimizer,
                       roc_auc=roc_auc,
                       train_loader=train_loader, train_writer=train_writer, val_loader=val_loader,
                       val_losses=val_losses,
                       val_writer=val_writer, log_dir=log_dir, fold=fold, criterion_vol_regr=criterion_vol_regr,
                       dataset_refresh_metadata=None, #(dataset, train_idx),
                       mixup_trainer=mixup_trainer, mixup_train_loader=mixup_train_loader,
                       contras_trainer=contras_trainer,
                       feature_alignment_loss_and_optim=[feature_alignment_loss, optimizer_centre_loss],
                       scheduler=scheduler)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()
        # Now we will load the best model in the given fold
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(log_dir, f"{model}_{fold}.pth"))))
        test_roc_auc.append(
            eval_roc_auc(model, test_loader, enc, epoch=fold, writer=None, criterion_vol_regr=criterion_vol_regr))
        test_regr_loss.append(eval_regr_loss(model, test_loader, criterion_vol_regr=criterion_vol_regr))
        # if test_writer is not None:
        #     acc, feat_split, gt_split = eval_acc(model, test_loader, criterion_vol_regr=criterion_vol_regr,
        #                                          writer=test_writer)
        #     test_accuracy.append(acc)
        #     out_feat = torch.cat((out_feat, feat_split), 0)
        #     outGT = torch.cat((outGT, gt_split), 0)
        # else:
        #     test_accuracy.append(
        #         eval_acc(model, test_loader, criterion_vol_regr=criterion_vol_regr, writer=test_writer))
        test_acc, size_cm_dict = eval_graph_len_acc(model, test_dataset, criterion_vol_regr=criterion_vol_regr)
        # assert np.allclose(test_acc, test_accuracy[-1]), f"Value differs as {test_acc} vs. {test_accuracy[-1]}"
        plot_results_based_on_graph_size(size_cm_dict, output_dir=log_dir, filename_acc=f'acc_{fold}',
                                         filename_roc=f'roc_{fold}', model_type=model, fold=fold)
        print(f"Test accuracy {test_acc}, Test ROC {test_roc_auc[-1]} and Test Regr loss {test_regr_loss[-1]}")
        # Doing the same for best loss model
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(log_dir, f"{model}_loss_{fold}.pth"))))
        test_roc_auc_min_loss.append(
            eval_roc_auc(model, test_loader, enc, epoch=fold, writer=None, criterion_vol_regr=criterion_vol_regr))
        test_regr_loss_min_loss.append(eval_regr_loss(model, test_loader, criterion_vol_regr=criterion_vol_regr))
        # if test_writer is not None:
        #     acc, feat_split, gt_split = eval_acc(model, test_loader, criterion_vol_regr=criterion_vol_regr,
        #                                          writer=test_writer)
        #     test_accuracy_min_loss.append(acc)
        #     out_feat = torch.cat((out_feat, feat_split), 0)
        #     outGT = torch.cat((outGT, gt_split), 0)
        # else:
        #     test_accuracy_min_loss.append(
        #         eval_acc(model, test_loader, criterion_vol_regr=criterion_vol_regr, writer=test_writer))
        test_acc, size_cm_dict = eval_graph_len_acc(model, test_dataset, criterion_vol_regr=criterion_vol_regr)
        # assert np.allclose(test_acc, test_accuracy[-1]), f"Value differs as {test_acc} vs. {test_accuracy[-1]}"
        print(
            f"Test accuracy {test_acc}, Test ROC {test_roc_auc_min_loss[-1]} and Test Regr loss {test_regr_loss_min_loss[-1]}")

    # test_writer.add_embedding(out_feat.cpu().numpy(), metadata=outGT.cpu().numpy().tolist())
    loss, roc_auc, test_roc_auc, duration, test_accuracy, test_regr = torch.as_tensor(val_losses), torch.as_tensor(
        roc_auc), torch.as_tensor(
        test_roc_auc), torch.as_tensor(durations), torch.as_tensor(test_accuracy), torch.as_tensor(test_regr_loss)
    loss, roc_auc = loss.view(folds, epochs), roc_auc.view(folds, epochs)
    # Select the best model based on val

    loss_mean = loss.mean().item()
    test_roc_auc_mean = test_roc_auc.mean().item()
    test_roc_auc_std = test_roc_auc.unsqueeze(0).std().item()
    test_accuracy_mean = test_accuracy.mean().item()
    test_accuracy_std = test_accuracy.unsqueeze(0).std().item()
    test_regr_mean = test_regr.mean().item()
    test_regr_std = test_regr.unsqueeze(0).std().item()
    # Doing the same for min_loss variant
    test_roc_auc_min_loss, test_accuracy_min_loss, test_regr_min_loss = torch.as_tensor(test_roc_auc_min_loss), \
                                                                        torch.as_tensor(
                                                                            test_accuracy_min_loss), torch.as_tensor(
        test_regr_loss_min_loss)
    # loss, roc_auc = loss.view(folds, epochs), roc_auc.view(folds, epochs)
    test_roc_auc_min_loss_mean = test_roc_auc_min_loss.mean().item()
    test_roc_auc_min_loss_std = test_roc_auc_min_loss.unsqueeze(0).std().item()
    test_accuracy_min_loss_mean = test_accuracy_min_loss.mean().item()
    test_accuracy_min_loss_std = test_accuracy_min_loss.unsqueeze(0).std().item()
    test_regr_min_loss_mean = test_regr_min_loss.mean().item()
    test_regr_min_loss_std = test_regr_min_loss.unsqueeze(0).std().item()
    duration_mean = duration.mean().item()
    print(f'Val Loss: {loss_mean:.4f}, Test AUROC: {test_roc_auc_mean:.3f} '
          f'± {test_roc_auc_std:.3f}, Test Regr: {test_regr_mean} ± {test_regr_std:.3f}'
          f'\n Min loss variant Test AUROC: {test_roc_auc_min_loss_mean:.3f} ± {test_roc_auc_min_loss_std:.3f},'
          f' Test Regr: {test_regr_min_loss_mean} ± {test_regr_min_loss_std:.3f}')

    return loss_mean, test_roc_auc_mean, test_roc_auc_std, test_accuracy_mean, test_accuracy_std, test_regr_mean, \
           test_regr_std, test_roc_auc_min_loss_mean, test_roc_auc_min_loss_std, test_accuracy_min_loss_mean, \
           test_accuracy_min_loss_std, test_regr_min_loss_mean, test_regr_min_loss_std


def main(model_type):
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
                      sample_graph_data=sample_graph_data)
    print(model.get_full_des())
    loss, roc, roc_std, acc, acc_std, regr, regr_std, roc_mil, roc_std_mil, acc_mil, acc_std_mil, regr_mil, regr_std_mil = final_cross_validation_with_val_set(
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
    desc = f'{roc:.3f} ± {roc_std:.3f}, accuracy {acc:.3f} ± {acc_std:.3f} and regr {regr:.3f} ± {regr_std:.3f}'
    desc += f'{roc_mil:.3f} ± {roc_std_mil:.3f}, accuracy {acc_mil:.3f} ± {acc_std_mil:.3f} and regr {regr_mil:.3f} ± {regr_std_mil:.3f}'
    results += [f'{model}: {desc}']
    print(f'--\n{results}')
    return loss, roc, roc_std, acc, acc_std, regr, regr_std, roc_mil, roc_std_mil, acc_mil, acc_std_mil, regr_mil, regr_std_mil


if __name__ == '__main__':
    seed_everything(seed=42)
    cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    model_types = get_configurations_dtype_string_list(section='TRAINING', key='MODEL_TYPES')
    for model_type in model_types:
        main(model_type=model_type)
