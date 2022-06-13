import argparse
import os
import torch
from torch_geometric import seed_everything
import ray

from environment_setup import get_configurations_dtype_string_list
# This line is important for raytune.
# It was unable to run properly in multiple-GPU setup
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from dataset.dataset_factory import get_dataset
from model_training.train_eval import cross_validation_with_val_set

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
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


def main():
    ray.init()
    print(f"Is cuda available: {torch.cuda.is_available()}")
    print(ray.cluster_resources())
    method_dict = {}
    conv_names = get_configurations_dtype_string_list(section='TRAINING', key='MODEL_TYPES')
    seed_everything(seed=42)
    dataset = get_dataset()
    # Determines how many samples from random grid search are made
    num_samples = 10
    sample_graph_data = dataset[0][0]
    for conv_name in conv_names:
        test_roc, test_roc_std = cross_validation_with_val_set(
            dataset=dataset,
            model_type=conv_name,
            folds=10,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr_decay_factor=args.lr_decay_factor,
            lr_decay_step_size=args.lr_decay_step_size,
            weight_decay=1e-5,
            logger=None,
            sample_graph_data=sample_graph_data,
            num_samples=num_samples
        )
        print(f"Final performance {test_roc} with stddev {test_roc_std}")
        method_dict[conv_name] = (test_roc, test_roc_std)
    for key, value in method_dict.items():
        print(f"{key} has {value}")


if __name__ == '__main__':
    main()
