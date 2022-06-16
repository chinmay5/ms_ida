import os

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from environment_setup import PROJECT_ROOT_DIR


class LogWriterWrapper(object):
    def __init__(self, summary_writer):
        self.summary_writer = summary_writer

    def add_scalar(self, *args, **kwargs):
        if self.summary_writer is not None:
            self.summary_writer.add_scalar(*args, **kwargs)


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

def read_configs():
    configs = RunTimeConfigs()
    # configs.RAW_METADATA_CSV =
