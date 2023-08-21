import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.tools import resize_volume


class Dataset(torch.utils.data.Dataset):

    def __init__(self,
                 mode="total"
                 ):
        # basic initialize
        self.mode = mode
        self.train_pairs = pickle.load(
            open('/mnt/data_jixie1/chwang/final_dataset/MRI/lineared/final_train_pairs.pkl', 'rb'))
        self.valid_pairs = pickle.load(
            open('/mnt/data_jixie1/chwang/final_dataset/MRI/lineared/final_valid_pairs.pkl', 'rb'))
        self.test_pairs = pickle.load(
            open('/mnt/data_jixie1/chwang/final_dataset/MRI/lineared/final_test_pairs.pkl', 'rb'))
        self.nacc_pairs = pickle.load(
            open('/mnt/data_jixie1/chwang/final_dataset/MRI/lineared/final_nacc_pairs.pkl', 'rb'))
        self.imgs = []

        # original data id: NC->0 AD->1 sMCI->3 pMCI->4

        if mode == "total_cn":
            for key in ['0']:
                self.imgs += self.train_pairs[key]
        elif mode == "total_ad":
            for key in ['1']:
                self.imgs += self.train_pairs[key]
        elif mode == "total_mci":
            for key in ['3', '4']:
                self.imgs += self.train_pairs[key]
        elif mode == "valid":
            for key in ['3', '4']:
                self.imgs += self.valid_pairs[key]
        elif mode == "test":
            for key in ['3', '4']:
                self.imgs += self.test_pairs[key]
        elif mode == "nacc":
            for key in ['3', '4']:
                self.imgs += self.nacc_pairs[key]

    def __getitem__(self, index):
        data_path = self.imgs[index][0]
        value = self.imgs[index][1]
        if self.mode != "valid" and self.mode != "test" and self.mode != "nacc":
            if value == 1:
                value = 2
            # combing sMCI and pMCI data as MCI
            elif value == 3 or value == 4:
                value = 1
        else:
            if value == 3:
                value = 0
            elif value == 4:
                value = 1

        original_np = np.load(data_path)
        A = torch.from_numpy(
            resize_volume(original_np, 128, 128, 128)).type(
            torch.FloatTensor)
        return A.unsqueeze(0), value

    def __len__(self):
        #  the length of dataset
        return len(self.imgs)
