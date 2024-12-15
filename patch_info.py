import numpy as np
# import yuvio
from tqdm import tqdm
import os
import cv2


def ceildiv(a, b):
    return -(a // -b)


def extract_patches(r, g, b, patch_size):
    patches = []
    h, w = r.shape
    ph, pw = patch_size

    for i in range(0, h, ph):
        for j in range(0, w, pw):
            if i + ph <= h and j + pw <= w:
                r_patch = r[i:i+ph, j:j+pw]
                g_patch = g[i:i+ph, j:j+pw]
                b_patch = b[i:i+ph, j:j+pw]

                P = np.stack((r_patch, g_patch, b_patch), axis=2)
                patches.append(P)

    return patches


class PatchInfo:
    
    def __init__(self, imagePath, name, patch_size):
        self.imagePath = imagePath
        self.name = name
        self.patch_size = patch_size





def image_to_patches(patchinfo: PatchInfo):
    
    inputImage = os.path.join(patchinfo.imagePath, patchinfo.name)
    patch_size = patchinfo.patch_size

    
    f = cv2.imread(inputImage)
    r = f[:, :, 0]
    g = f[:, :, 1]
    b = f[:, :, 2]

    patches = extract_patches(r, g, b, (patchinfo.patch_size, patchinfo.patch_size))


    return patches