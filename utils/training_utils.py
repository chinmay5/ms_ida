import numpy as np
from sklearn.preprocessing import OneHotEncoder


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

