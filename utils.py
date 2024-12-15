import torch
import cv2
import numpy as np
import os

def saveTrainPatch(outputs, idx):
    outputs = torch.permute(outputs[15, :, :, :].squeeze(0), (1, 2, 0))
    R = outputs[:, :, 0].cpu().detach().numpy()
    G = outputs[:, :, 1].cpu().detach().numpy()
    B = outputs[:, :, 2].cpu().detach().numpy()

    HRImg = np.zeros([256, 256, 3])

    HRImg[:, :, 0] = R * 255.0
    HRImg[:, :, 1] = G * 255.0
    HRImg[:, :, 2] = B * 255.0

    target_folder = "./train_out_VidGame"
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    target_file_name = target_folder + '/patch' + str(idx) + '.png'
    cv2.imwrite(target_file_name, HRImg)
