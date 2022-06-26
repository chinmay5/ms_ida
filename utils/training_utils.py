import os

import numpy as np
import torch
from sklearn.preprocessing import OneHotEncoder

from environment_setup import PROJECT_ROOT_DIR


class LogWriterWrapper(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def add_scalar(self, *args, **kwargs):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(*args, **kwargs)


class CustomDictKey(object):
    def __init__(self, key_name, key_iden):
        super(CustomDictKey, self).__init__()
        self.key_name = key_name
        self.key_iden = key_iden

    def __eq__(self, other):
        return isinstance(other, CustomDictKey) and \
               other.key_name == self.key_name and \
               other.key_iden == self.key_iden

    def __hash__(self):
        return hash((self.key_name, self.key_iden))

    def __repr__(self):
        return f'{self.key_name} - {self.key_iden}'


class LabelEncoder(object):

    def __init__(self):
        enc = OneHotEncoder()
        possible_labels = np.array([0, 1]).reshape(-1, 1)
        enc.fit(possible_labels)
        self.encoder = enc

    def __call__(self):
        return self.encoder


class RunTimeConfigs(object):
    def __init__(self):
        self.configs = []

    def write_to_disk(self):
        base_log_dir = os.path.join(PROJECT_ROOT_DIR, self.logdir)
        os.makedirs(base_log_dir, exist_ok=True)
        filename = os.path.join(base_log_dir, "configs_for_run.cfg")
        with open(filename, 'w') as configfile:
            for config, value in vars(self):
                configfile.write(f"{config}: {value} \n")


def drop_nodes(data):
    # We can drop 30% of the nodes at random.
    node_mask = torch.rand(data.num_nodes) > 0.3
    data = data.subgraph(node_mask)
    return data
