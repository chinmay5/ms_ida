import argparse

import argparse
import os

import torch.utils.data
import torchio as tio
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from tqdm import tqdm

from ablation.full_volume_dataset import FullVolDataset
from ablation.inception_net_3d import InceptionModel, weight_reset
from ablation.resnet_3d import generate_model
from environment_setup import device, PROJECT_ROOT_DIR, get_configurations_dtype_string
from utils.training_utils import LabelEncoder, k_fold, get_class_weights, seed_worker, g, \
    LogWriterWrapper, display_mean_std, compute_precision_recall_acc_and_f1


def train(trainloader, optimizer, model, criterion, writer, epoch):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels, gt_regr = data
        # put the tensors to gpu
        inputs, labels, gt_regr = inputs.to(device), labels.to(device), gt_regr.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs, regr = model(inputs)
        loss = criterion(outputs, labels)
        # loss += F.mse_loss(gt_regr, regr.squeeze())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    writer.add_scalar('train_loss', running_loss / len(trainloader), global_step=epoch)


def eval_loss_and_auroc(loader, model, criterion, writer, epoch, enc):
    model.eval()
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, gt_regr = data
            # put the tensors to gpu
            inputs, labels, gt_regr = inputs.to(device), labels.to(device), gt_regr.to(device)
            outputs, regr = model(inputs)
            loss = criterion(outputs, labels)
            # Adding the regression loss
            # loss += F.mse_loss(gt_regr, regr.squeeze())
            running_loss += loss.item()
            outPRED = torch.cat((outPRED, outputs), 0)
            outGT = torch.cat((outGT, labels), 0)
    # Now we can compute the ROC
    predictions = torch.softmax(outPRED, dim=1)
    predictions, target = predictions.cpu().numpy(), outGT.cpu().numpy()
    # Encoder is callable.
    # Hence, we execute callable which returns the self.encoder instance
    target_one_hot = enc().transform(target.reshape(-1, 1)).toarray()  # Reshaping needed by the library
    # Arguments take 'GT' before taking 'predictions'
    roc_auc_value = roc_auc_score(target_one_hot, predictions, average='weighted')
    writer.add_scalar('loss', running_loss / len(loader))
    writer.add_scalar('roc', roc_auc_value, global_step=epoch)
    prec, recall, acc, f1 = compute_precision_recall_acc_and_f1(y_true=target, y_pred=predictions)
    return running_loss / len(loader), roc_auc_value, prec, recall, acc, f1


def cnn_balanced_batch_sampler(train_dataset):
    per_sample_wt = [0] * len(train_dataset)
    class_weights = get_class_weights(train_dataset)
    for idx, (graph, label, reg) in enumerate(train_dataset):
        class_weight = class_weights[label]
        per_sample_wt[idx] = class_weight  # Assigning the length weight to our sampler
    weighted_sampler = WeightedRandomSampler(per_sample_wt, num_samples=len(per_sample_wt), replacement=True)
    return weighted_sampler


def train_eval_model(epochs, dataset, folds, weight_decay, args):
    log_dir = os.path.join(PROJECT_ROOT_DIR,
                           get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    test_roc_min_loss = []
    prec_list_min_loss, recall_list_min_loss, acc_list_min_loss, f1_list_min_loss = [], [], [], []
    batch_size, lr = args.batch_size, args.lr

    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(dataset, folds))):
        if args.model == 'inception':
            print("Building inception model")
            model = InceptionModel(in_channel=2, block1_out_ch=16, block2_out_ch=32, block3_out_ch=64, vol_size=144)
        else:
            print("building resnet model")
            model = generate_model(model_depth=10, n_classes=2, n_input_channels=2)

        # We take augmented volume for the training split
        train_dataset = [(dataset[idx.item()][1], dataset[idx.item()][2], dataset[idx.item()][3]) for idx in train_idx]
        test_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2], dataset[idx.item()][3]) for idx in test_idx]
        val_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2], dataset[idx.item()][3]) for idx in val_idx]
        class_balance_weights = get_class_weights(train_dataset)
        balanced_sampler = cnn_balanced_batch_sampler(train_dataset=train_dataset)
        # Creating the data loaders
        print(f"Class balance weight is {class_balance_weights}")
        train_loader = DataLoader(train_dataset, batch_size, sampler=balanced_sampler, worker_init_fn=seed_worker,
                                  generator=g)
        # train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        # BEGIN: Logger creation
        os.makedirs(log_dir, exist_ok=True)
        train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'train_{fold}')))
        val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'val_{fold}')))
        enc = LabelEncoder()
        # Begin training
        min_loss = float("inf")
        best_model_save_epoch = -1
        model.to(device)
        for epoch in tqdm(range(epochs)):
            train(trainloader=train_loader, optimizer=optimizer, model=model, criterion=criterion, writer=train_writer,
                  epoch=epoch)
            # Also compute roc and loss for the training set.
            # This would help us in figuring out an overfitting scenario.
            train_loss, train_roc, _, _, _, _ = eval_loss_and_auroc(loader=train_loader, model=model,
                                                                    criterion=criterion,
                                                                    writer=train_writer, epoch=epoch, enc=enc)
            val_loss, val_roc, _, _, _, _ = eval_loss_and_auroc(loader=val_loader, model=model, criterion=criterion,
                                                                writer=val_writer, epoch=epoch, enc=enc)
            if val_loss < min_loss:
                best_model_save_epoch = epoch
                min_loss = val_loss
                torch.save(model.state_dict(), os.path.join(log_dir, f"{args.model}_loss_{fold}.pth"))
        # Let us free up the memory
        del train_dataset, val_dataset, test_dataset
        print(f"Training finished for fold: {fold}")
        print(f"Best model saved at epoch: {best_model_save_epoch}")

        model.apply(weight_reset)
        print(model.load_state_dict(torch.load(os.path.join(log_dir, f"{args.model}_loss_{fold}.pth"))))
        # We evaluate results on the test split now.
        test_loss, test_roc, prec_min_loss, recall_min_loss, acc_min_loss, f1_min_loss = eval_loss_and_auroc(
            loader=test_loader, model=model, criterion=criterion,
            writer=LogWriterWrapper(None), epoch=0, enc=enc)
        test_roc_min_loss.append(test_roc)
        prec_list_min_loss.append(prec_min_loss)
        recall_list_min_loss.append(recall_min_loss)
        acc_list_min_loss.append(acc_min_loss)
        f1_list_min_loss.append(f1_min_loss)
    # The evaluation is finished. Now we finalize the results

    test_roc_min_loss = torch.as_tensor(test_roc_min_loss)
    prec_tensor_min_loss, recall_tensor_min_loss, acc_tensor_min_loss, f1_tensor_min_loss = torch.as_tensor(prec_list_min_loss), torch.as_tensor(recall_list_min_loss), \
                                                                                          torch.as_tensor(acc_list_min_loss), torch.as_tensor(f1_list_min_loss)
    test_roc_min_loss_mean = test_roc_min_loss.mean().item()
    test_roc_min_loss_std = test_roc_min_loss.unsqueeze(0).std().item()
    print(f'Test AUROC: {test_roc_min_loss_mean:.3f} ± {test_roc_min_loss_std:.3f}')
    # Now the same for other metrics
    display_mean_std(metric_name='prec', metric_value_tensor=prec_tensor_min_loss)
    display_mean_std(metric_name='recall', metric_value_tensor=recall_tensor_min_loss)
    display_mean_std(metric_name='acc', metric_value_tensor=acc_tensor_min_loss)
    display_mean_std(metric_name='f1-score', metric_value_tensor=f1_tensor_min_loss)


if __name__ == '__main__':
    seed_everything(seed=42)
    cudnn.benchmark = False
    #  Can't use since max_pool 3d is non-deterministic
    # torch.use_deterministic_algorithms(True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3 * 1e-4)
    parser.add_argument('--model', type=str, default='inception')
    args = parser.parse_args()
    num_folds = 10
    weight_decay = 1e-4
    transform = tio.Compose([
        tio.RandomAffine(),
        tio.RandomBlur(),
        tio.RandomNoise(std=0.1),
        tio.RandomGamma(log_gamma=(-0.3, 0.3))
    ])
    dataset = FullVolDataset(transform=transform)
    train_eval_model(epochs=args.epochs, dataset=dataset, folds=num_folds, args=args,
                     weight_decay=weight_decay)
