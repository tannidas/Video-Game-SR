import os
import torch
import numpy as np
import torch.utils.data as data
import h5py
import random
import cv2

def read_data_at_index(file_path, index):
    with h5py.File(file_path, 'r') as hf:
        group_name = f"{index}"
        if group_name in hf:
            data_entry = {
                "HR": np.array(hf[group_name]["HR"]),
                "LR": np.array(hf[group_name]["LR"])
            }
            return data_entry
        else:
            print(f"No data found at index {index}")
            return None


class trainDataSet(data.Dataset):
    def __init__(self, train_dataset_file):

        
        self.file = train_dataset_file
        self.pickKeys = []

        with h5py.File(self.file, "r") as f:

            keys = f.keys()

            self.pickKeys += keys #random.sample(keys, k=200000)
        

        with open("Output.txt", "w") as text_file:
            for key in self.pickKeys:
                text_file.write(str(key)+"\n")

    
    def __len__(self):

        totalKeys = len(self.pickKeys)

        return totalKeys

    def __getitem__(self, index):

        data = read_data_at_index(self.file, self.pickKeys[index])

        hr_image = data["HR"].astype(np.float32) / 255.0
        lr_image = data["LR"].astype(np.float32) / 255.0

        hr_image = hr_image.transpose(2, 0, 1).astype(np.float32)
        lr_image = lr_image.transpose(2, 0, 1).astype(np.float32)


        hr_image = torch.from_numpy(hr_image).float()
        lr_image = torch.from_numpy(lr_image).float()

        return hr_image, lr_image




def getTestLRImage(fileName):

    image = cv2.imread(fileName)

    lr_image = image.astype(np.float32) / 255.0

    lr_image = lr_image.transpose(2, 0, 1).astype(np.float32)
    lr_image = np.expand_dims(lr_image, axis=0)

    lr_image = torch.from_numpy(lr_image).float()
    return lr_image