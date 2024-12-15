import base64
import zlib

import cv2
import numpy as np
import pandas as pd
import os


def encode(img: np.ndarray) -> bytes:
    """
    Lossless encoding of images for submission on kaggle platform.

    Parameters
    ----------
    img : np.ndarray
        cv2.imread(f) - BGR image in (h, w, c) format, c = 3 (even for png format).

    Returns
    -------
    bytes
        Encoded image as bytes.
    """
    img_to_encode = img.astype(np.uint8)
    img_to_encode = img_to_encode.flatten()
    img_to_encode = np.append(img_to_encode, -1)

    cnt, rle = 1, []
    for i in range(1, img_to_encode.shape[0]):
        if img_to_encode[i] == img_to_encode[i - 1]:
            cnt += 1
            if cnt > 255:
                rle += [img_to_encode[i - 1], 255]
                cnt = 1
        else:
            rle += [img_to_encode[i - 1], cnt]
            cnt = 1

    compressed = zlib.compress(bytes(rle), zlib.Z_BEST_COMPRESSION)
    base64_bytes = base64.b64encode(compressed)
    return base64_bytes

if __name__ == "__main__":

    finalResult = 'submission.csv'
    imgDir = "./Network_Result"
    listOfImages = os.listdir(imgDir)

    dataDict = dict()
    dataDict["id"] = []
    dataDict["filename"] = []
    dataDict["rle"] = []

    for id, imgName in enumerate(listOfImages):

        fullPath = os.path.join(imgDir, imgName)

        img = cv2.imread(fullPath)
        encoded_img = str(encode(img))

        dataDict["id"].append(id)
        dataDict["filename"].append(imgName)
        dataDict["rle"].append(encoded_img)

        print("{}% Completed".format((id / len(listOfImages))*100))
    
    df = pd.DataFrame(dataDict)
    df.to_csv(finalResult, index=False)