import h5py
from patch_info import *


hrImagePath = "/data4/super-resolution-in-video-games/train/hr"
lrImagePath = "/data4/super-resolution-in-video-games/train/lr"
allOriginalPatchSaveLoc = "/data3/ALLPatches_Vid_Game.h5"
hrPatchSize = 256
lrPatchSize = 64


list_of_images = os.listdir(lrImagePath)



with h5py.File(allOriginalPatchSaveLoc, 'w') as hf:

    grpNum = 0

    for img in tqdm(list_of_images):

        hrPatches = PatchInfo(hrImagePath, img, hrPatchSize)
        lrPatches = PatchInfo(lrImagePath, img, lrPatchSize)

        allHRPatches = image_to_patches(hrPatches)
        allLRPatches = image_to_patches(lrPatches)
            

        TotalHRPatches = len(allHRPatches)
        TotalLRPatches = len(allLRPatches)

        
        print("Total {} patches created from {}".format(TotalHRPatches, img))
        print("Total {} patches created from {}".format(TotalLRPatches, img))

        for i, (hrPatch, lrPatch) in enumerate(zip(allHRPatches, allLRPatches)):
            grp = hf.create_group(f"{grpNum}")
            grp.create_dataset(f"HR", data=hrPatch)  # Create dataset for original patches
            grp.create_dataset(f"LR", data=lrPatch)  # Create dataset for reconstructed patches
            grpNum += 1