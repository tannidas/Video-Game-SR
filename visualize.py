import numpy as np

import h5py
import cv2


def calcTotalData(filePath):

    with h5py.File(filePath, "r") as f:
        keys = f.keys()
        totalKeys = len(keys)

    return totalKeys

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

# Example usage
file_path = '/data3/ALLPatches_DVI2K.h5'
totalKeys = calcTotalData(file_path)

index = input("Enter a number between 0-{} to visualize: ".format(totalKeys))

data_entry = read_data_at_index(file_path, index)

if data_entry is not None:
    print("Org:", data_entry["HR"].shape)
    print("Rec:", data_entry["LR"].shape)

HRImg = np.zeros([256, 256, 3])

HRImg[:, :, 0] = data_entry["HR"][:, :, 0]
HRImg[:, :, 1] = data_entry["HR"][:, :, 1]
HRImg[:, :, 2] = data_entry["HR"][:, :, 2]

cv2.imwrite("HR_{}.png".format(index), HRImg)

LRImg = np.zeros([64, 64, 3])

LRImg[:, :, 0] = data_entry["LR"][:, :, 0]
LRImg[:, :, 1] = data_entry["LR"][:, :, 1]
LRImg[:, :, 2] = data_entry["LR"][:, :, 2]


cv2.imwrite("LR_{}.png".format(index), LRImg)