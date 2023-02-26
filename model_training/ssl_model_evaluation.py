import argparse
import numpy as np
import os
import time
import torch
from torch import nn
from torch.backends import cudnn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAE
from tqdm import tqdm

from dataset.dataset_factory import get_dataset
from environment_setup import write_configs_to_disk, PROJECT_ROOT_DIR, \
    get_configurations_dtype_string, get_configurations_dtype_string_list, device, get_configurations_dtype_int
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
from graph_models.homogeneous_models import SSLGCNEncoder
from graph_models.model_factory import get_model
from utils.training_utils import LogWriterWrapper, get_dataset_and_auxiliary_loss, k_fold
import torch_geometric.transforms as T

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay_factor', type=float, default=0.5)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
args = parser.parse_args()


def logger(info):
    fold, epoch = info['fold'] + 1, info['epoch']
    val_loss, test_acc = info['val_loss'], info['test_acc']
    print(f'{fold:02d}/{epoch:03d}: Val Loss: {val_loss:.4f}, '
          f'Test Accuracy: {test_acc:.3f}')


@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)


def train(model, train_dataset, optimizer, writer, epoch):
    model.train()
    for idx, (graph) in enumerate(train_dataset):
        # Skip for isolated graphs
        if graph.pos_edge_label_index.shape[1] == 0:
            continue
        optimizer.zero_grad()
        z = model.encode(graph)
        loss = model.recon_loss(z, graph.pos_edge_label_index)
        # if args.variational:
        #     loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
        loss.backward()
        optimizer.step()
        writer.add_scalar('ssl_loss', loss.item(), global_step=epoch * len(train_dataset) + idx)


def extract_ssl_features(model, dataset, epoch, fold, logdir, is_train=True):
    features = []
    feature_type = 'train' if is_train else 'test'
    for idx, (graph, _) in enumerate(dataset):
        graph = graph.to(device)
        z = model.encode(graph)
        features.append(z.detach().cpu().numpy())
    features = np.concatenate(features)
    folder_base = os.path.join(logdir, 'ssl_features', str(epoch))
    os.makedirs(folder_base, exist_ok=True)
    np.save(os.path.join(folder_base, f'{feature_type}_ssl_{fold}'), features)


def ssl_training_and_feature_extraction(dataset, model, folds, batch_size, lr, weight_decay, num_layers, hidden, args, ssl_transform):
    # For PCA visualization
    # Changing this piece to see the PCA
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    log_dir = os.path.join(base_log_dir, f"_ssl_layers_{num_layers}_hidden_dim_{hidden}")
    for fold, (train_idx, val_idx,
               test_idx) in enumerate(zip(*k_fold(dataset, folds))):

        _, test_dataset, train_dataset, val_dataset = get_dataset_and_auxiliary_loss(dataset,
                                                                                     test_idx,
                                                                                     train_idx,
                                                                                     val_idx)
        # Since we are working in SSL way, we just combine the train and val splits together.
        # The test split is used to extract features
        combined_dataset = train_dataset.copy()
        combined_dataset.extend(val_dataset)
        # Now apply the transform to ensure we get positive and negative links.
        # These can be used to train the model
        ssl_link_pred_train_dataset = generate_pos_neg_links(ssl_transform, combined_dataset)
        # Since we are only interested in the features, we can use
        train_loader = DataLoader(ssl_link_pred_train_dataset, batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
        print(
            f"Starting training with train size:\nTrain: {len(ssl_link_pred_train_dataset)}\nTest: {len(test_dataset)}")

        model.to(device).reset_parameters()
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        os.makedirs(log_dir, exist_ok=True)
        train_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'ssl_train_{fold}')))
        val_writer = LogWriterWrapper(SummaryWriter(os.path.join(log_dir, f'ssl_val_{fold}')))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        for epoch in tqdm(range(args.epochs)):
            train(model=model, train_dataset=ssl_link_pred_train_dataset, optimizer=optimizer, writer=train_writer, epoch=epoch)
            # Save model weights and the extracted features
            if epoch % 100 == 99 or epoch == args.epochs - 1:
                # Save the model state
                torch.save(model.state_dict(), os.path.join(log_dir, f"{model}_{fold}.pth"))
                extract_ssl_features(model=model, dataset=test_dataset, epoch=epoch, fold=fold, logdir=log_dir, is_train=False)
                extract_ssl_features(model=model, dataset=train_dataset, epoch=epoch, fold=fold, logdir=log_dir, is_train=True)


def generate_pos_neg_links(ssl_transform, train_dataset):
    result_dataset = []
    for x, _ in train_dataset:
        try:
            # Only add the training split. The val and test splits have no edges.
            modified_dataset = ssl_transform(x)[0]
        except:
            x["pos_edge_label_index"] = torch.tensor([[], []], dtype=torch.long)
            modified_dataset = x.clone()
        result_dataset.append(modified_dataset)
    return result_dataset


def main():
    ssl_transform = T.Compose([
        T.NormalizeFeatures(),
        T.ToDevice(device),
        T.ToUndirected(),
        T.RandomLinkSplit(num_val=0, num_test=0, is_undirected=True,
                          split_labels=True, add_negative_train_samples=False),
    ])
    dataset = get_dataset()
    # Write configurations to the disk
    # Please do this from the main process.
    # Once we do it, the initial set of configs are persisted
    write_configs_to_disk()
    # Determines how many samples from random grid search are made
    num_folds = 10
    num_layers = 2
    hidden = 64
    print(dataset)
    node_feature_dim = get_configurations_dtype_int(section='TRAINING', key='NODE_FEAT_DIM')
    model = GAE(SSLGCNEncoder(node_feature_dim, hidden_dim=hidden))

    print(model)
    ssl_training_and_feature_extraction(
        dataset=dataset,
        model=model,
        folds=num_folds,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-5,
        num_layers=num_layers,
        hidden=hidden,
        args=args,
        ssl_transform=ssl_transform
    )


if __name__ == '__main__':
    seed_everything(seed=42)
    cudnn.benchmark = True
    main()
