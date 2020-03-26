import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import os


class DataSets(Dataset):
    def __init__(self, path, data="iemocap", split_type="train", if_align=False):
        super(DataSets, self).__init__()
        path = os.path.join(path, data + "_data.pkl" if if_align else data + "_data_noalign.pkl")
        with open(path, "rb") as f:
            dataset = pickle.load(f)

        # These are torch tensors
        self.vision = torch.tensor(dataset[split_type]["vision"].astype(np.float32)).cpu().detach()
        self.text = torch.tensor(dataset[split_type]["text"].astype(np.float32)).cpu().detach()
        self.audio = dataset[split_type]["audio"].astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(dataset[split_type]["labels"].astype(np.float32)).cpu().detach()

        # Note: this is STILL an numpy array
        self.meta = dataset[split_type]["id"] if "id" in dataset[split_type].keys() else None

        self.data = data

        self.n_modalities = 3  # vision/ text/ audio

    def get_n_modalities(self):
        return self.n_modalities

    def get_seq_len(self):
        return self.text.shape[1], self.audio.shape[1], self.vision.shape[1]

    def get_dim(self):
        return self.text.shape[2], self.audio.shape[2], self.vision.shape[2]

    def get_lbl_info(self):
        # return number_of_labels, label_dim
        return self.labels.shape[1], self.labels.shape[2]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = (index, self.text[index], self.audio[index], self.vision[index])
        y = self.labels[index]
        meta = (
            (0, 0, 0)
            if self.meta is None
            else (self.meta[index][0], self.meta[index][1], self.meta[index][2])
        )
        if self.data == "mosi":
            meta = (
                self.meta[index][0].decode("UTF-8"),
                self.meta[index][1].decode("UTF-8"),
                self.meta[index][2].decode("UTF-8"),
            )
        if self.data == "iemocap":
            y = torch.argmax(y, dim=-1)
        return x, y, meta
