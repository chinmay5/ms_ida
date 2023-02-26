import time

import argparse
import os
import torch
from torch_geometric import seed_everything
import ray

from environment_setup import get_configurations_dtype_string_list, write_configs_to_disk

# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataset.dataset_factory import get_dataset
from model_training.train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=500)
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


def main():
    start_time = time.time()
    method_dict = {}
    conv_names = get_configurations_dtype_string_list(section='TRAINING', key='MODEL_TYPES')
    seed_everything(seed=42)
    dataset = get_dataset()
    # Write configurations to the disk
    # Please do this from the main process.
    # Once we do it, the initial set of configs are persisted
    write_configs_to_disk()
    # Determines how many samples from random grid search are made
    num_samples = 5
    sample_graph_data = dataset[0][0]
    # Initialize raytune.
    # Doing this after storing the configs is important.
    # Otherwise, the updated env variable is not copied to each individual process.
    # Thus, we can not start multiple processes in that scenario.
    ray.init()
    print(f"Is cuda available: {torch.cuda.is_available()}")
    print(ray.cluster_resources())
    for conv_name in conv_names:
        loss, roc, roc_std, acc, acc_std, regr, regr_std = cross_validation_with_val_set(
            dataset=dataset,
            model_type=conv_name,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=0,
            logger=None,
            sample_graph_data=sample_graph_data,
            num_samples=num_samples
        )
        desc = f'{roc:.3f} ± {roc_std:.3f}, accuracy {acc:.3f} ± {acc_std:.3f} and regr {regr:.3f} ± {regr_std:.3f}'
        method_dict[conv_name] = desc
    for key, value in method_dict.items():
        print(f"{key} has {value}")
    print(f"time taken is {(time.time() - start_time)/3600} hours")


if __name__ == '__main__':
    main()
