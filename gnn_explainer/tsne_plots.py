import argparse
import os
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader

from environment_setup import PROJECT_ROOT_DIR, \
    get_configurations_dtype_string, get_configurations_dtype_boolean
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
from graph_models.model_factory import get_model
from model_training.train_eval import k_fold
from utils.training_utils import LogWriterWrapper
from matplotlib import pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataset.dataset_factory import get_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
args = parser.parse_args()


def generate_tsne_embedding(dataset, model, folds, batch_size,
                            split_acc_based_on_labels, device, checkpoint_dir):
    outGT = torch.FloatTensor().to(device)
    outPRED = torch.FloatTensor().to(device)
    base_log_dir = os.path.join(PROJECT_ROOT_DIR,
                                get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'))
    logger = LogWriterWrapper(SummaryWriter(os.path.join(base_log_dir, f'test')))
    test_images_dir = os.path.join(base_log_dir, 'final_test_images')
    os.makedirs(test_images_dir, exist_ok=True)
    for fold, (_, _, test_idx) in enumerate(zip(*k_fold(dataset, folds))):
        is_node_level_dataset = get_configurations_dtype_boolean(section='SETUP', key='PERFORM_NODE_LEVEL_PREDICTION')
        if is_node_level_dataset:
            test_dataset = [(dataset[idx.item()][0], dataset[idx.item()][2]) for idx in test_idx]
            criterion_vol_regr = nn.L1Loss()
        else:
            test_dataset = [dataset[idx.item()] for idx in test_idx]
            criterion_vol_regr = None
        # We also need to obtain class weights to ensure we do not have data imbalance issues.
        test_loader = DataLoader(test_dataset, batch_size, shuffle=False, pin_memory=True, num_workers=4)
        # Let us also create the One Hot Encoder needed for auroc computation

        model.to(device).reset_parameters()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # We will load the best model in the given fold.
        # The checkpoints are already saved. We would simply load the values
        model.reset_parameters()
        print(model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{model}_{fold}.pth"))))

        model.eval()
        is_node_and_graph_cls = criterion_vol_regr is not None
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            with torch.no_grad():
                if is_node_and_graph_cls:
                    pred, _, _ = model(data)  # Ignoring the node & regr component for the time being
                else:
                    pred = model(data)
            outPRED = torch.cat((outPRED, pred), 0)
            outGT = torch.cat((outGT, labels), 0)
    features = outPRED.cpu()
    labels = outGT.cpu().numpy()
    logger.add_embedding(mat=features)

    tsne = TSNE(n_components=2, random_state=0)
    X_2d = tsne.fit_transform(features.numpy())

    plt.figure(figsize=(6, 5))
    colors = 'r', 'g'
    target_ids = [0, 1]
    target_names =[0, 1]
    for i, c, label in zip(target_ids, colors, target_names):
        plt.scatter(X_2d[labels == i, 0], X_2d[labels == i, 1], c=c, label=label)
    plt.legend()
    plt.show()


def main(checkpoint_dir, model_type, hidden, num_layers, split_acc_based_on_labels, report_cm):
    seed_everything(seed=42)
    dataset = get_dataset()
    # Determines how many samples from random grid search are made
    num_folds = 10
    sample_graph_data = dataset[0][0]

    # Check if cuda available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = get_dataset()
    model = get_model(model_type=model_type, hidden_dim=hidden, num_layers=num_layers,
                      sample_graph_data=sample_graph_data)
    print(model.get_full_des())
    generate_tsne_embedding(
        dataset=dataset,
        model=model,
        folds=num_folds,
        batch_size=args.batch_size,
        split_acc_based_on_labels=split_acc_based_on_labels,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == '__main__':
    hidden = 256
    num_layers = 2
    model_type = get_configurations_dtype_string(section='TRAINING', key='MODEL_TYPES')
    checkpoint_dir = os.path.join(PROJECT_ROOT_DIR,
                                  get_configurations_dtype_string(section='TRAINING', key='LOG_DIR'),
                                  f'_layers_{num_layers}_hidden_dim_{hidden}'
                                  )

    main(checkpoint_dir=checkpoint_dir, model_type=model_type, hidden=hidden, num_layers=num_layers,
         split_acc_based_on_labels=True, report_cm=True)
