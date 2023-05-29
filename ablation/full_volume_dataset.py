import glob
import os

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torchio as tio
from torch.utils.data import Dataset, DataLoader

from environment_setup import get_configurations_dtype_boolean, get_configurations_dtype_string
from utils.dataset_util import convert_to_pickle_file_name

# Create a folder that contains a combination of these two factors.
full_vol_dataset_base_path = get_configurations_dtype_string(section='SETUP', key='FULL_VOL_DATASET_BASE_PATH')


class FullVolDataset(Dataset):
    def __init__(self, transform):
        super(FullVolDataset, self).__init__()
        use_2y = get_configurations_dtype_boolean(section='SETUP', key='USE_2Y')
        if use_2y:
            self.annotated_data_pickle_location = convert_to_pickle_file_name(
                get_configurations_dtype_string(section='SETUP',
                                                key='ANNOTATED_DATA_2Y_CSV_LOCATION'))
            self.label_column = 'New_Lesions_2y_Label'
            self.graph_regr_column = 'New_Lesions_2y_volume_mm3'
            print("using 2y labels")
        else:
            self.annotated_data_pickle_location = convert_to_pickle_file_name(
                get_configurations_dtype_string(section='SETUP',
                                                key='ANNOTATED_DATA_1Y_CSV_LOCATION'))
            self.label_column = 'New_Lesions_1y_Label'
            self.graph_regr_column = f'New_Lesions_1y_volume_mm3'
        # Now we can calculate the required dataset attributes
        annotated_data = pd.read_pickle(self.annotated_data_pickle_location)
        self.patient_list = annotated_data.loc[:, 'Patient']
        self.y = annotated_data.loc[:, self.label_column]
        self.y_regr = annotated_data.loc[:, self.graph_regr_column]
        self.transform = transform
        self.resize_transform = tio.CropOrPad((144, 144, 144))

    def __len__(self):
        return len(self.patient_list)

    def __getitem__(self, item):
        patient_name = f"sub-{self.patient_list[item]}"
        patient_label = int(self.y[item].item())
        graph_vol = torch.as_tensor(self.y_regr[item].item(), dtype=torch.float)
        # self.generate_concatenated_scan_volume(patient_name=patient_name)
        volume = self.load_volume(patient_name=patient_name)
        augmented_volume = self.transform(volume)
        volume, augmented_volume = self._min_max_normalize_data(volume), self._min_max_normalize_data(augmented_volume)
        volume, augmented_volume = self.resize_transform(volume), self.resize_transform(augmented_volume)
        return volume, augmented_volume, patient_label, graph_vol

    def generate_concatenated_scan_volume(self, patient_name):
        """
        This function should be called only once. It will go through the patient folder and concatenate T1 and Flair scans.
        This concatenated scan will be saved for later usage.
        :param patient_name: Name of the patient
        :return: None
        """
        flair_filename = glob.glob(f"{full_vol_dataset_base_path}/{patient_name}/**/*mni_flair.nii.gz")[0]
        t1_filename = glob.glob(f"{full_vol_dataset_base_path}/{patient_name}/**/*mni_t1.nii.gz")[0]
        # Let us also get the masks for each of the scans

        # Now we need to convert the nifti file into numpy array
        flair = np.array(nib.load(flair_filename).dataobj)
        t1 = np.array(nib.load(t1_filename).dataobj)
        volume = np.stack([t1, flair])
        np.save(file=os.path.join(full_vol_dataset_base_path, f'{patient_name}_scan.npy'), arr=volume)

    def load_volume(self, patient_name):
        return torch.as_tensor(np.load(file=os.path.join(full_vol_dataset_base_path, f'{patient_name}_scan.npy')),
                               dtype=torch.float)

    def _min_max_normalize_data(self, volume):
        max_val, min_val = volume.max(), volume.min()
        volume = (volume - min_val) / (max_val - min_val)
        return volume


if __name__ == '__main__':
    dataset = FullVolDataset(transform=tio.RandomAffine())
    loader = DataLoader(dataset, 16)
    print(dataset)
    print(len(dataset))
    for data in loader:
        print(data[0].shape)
        print(data[0].dtype)
        print(data[-1].dtype)
