import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from dataset import *
from testNet_5 import *
import yuvio as io
from torch.nn.functional import interpolate
import cv2

cuda_no = 2
trained_model_loc = "./TestNet5_Models/TestNet5_VidGame_ep_30.pt"
test_dataset_folder = "/data4/super-resolution-in-video-games/test/lr"
network_result_dir = "Network_Result"


def Var(x):
    return Variable(x.cuda(cuda_no), requires_grad = False)


net = TestNet_5(3, 16, True).cuda(cuda_no)
net.load_state_dict(torch.load(trained_model_loc))



for param in net.parameters():
    param.requires_grad = False


listOfImages = os.listdir(test_dataset_folder)

idx = 1

for img in listOfImages:

    imgPath = os.path.join(test_dataset_folder, img)
    lr_image = getTestLRImage(imgPath)  


    with torch.no_grad():

        lr_input_image = Var(lr_image)


        output_hr_image = net(lr_input_image)
        output_hr_image = output_hr_image.squeeze(0)

        output_hr_image = output_hr_image.clamp(0., 1.)

        output_hr_image = torch.permute(output_hr_image, (1, 2, 0))

        R = output_hr_image[:, :, 0].cpu().detach().numpy()
        G = output_hr_image[:, :, 1].cpu().detach().numpy()
        B = output_hr_image[:, :, 2].cpu().detach().numpy()

        
        HRImg = np.zeros([256, 256, 3])

        HRImg[:, :, 0] = R * 255.0
        HRImg[:, :, 1] = G * 255.0
        HRImg[:, :, 2] = B * 255.0
        

        target_folder = os.path.join(network_result_dir)

        if not os.path.exists(target_folder):
            os.makedirs(target_folder)

        target_file_name = os.path.join(network_result_dir, img)

        cv2.imwrite(target_file_name, HRImg)

        print("{}% Completed".format((idx / len(listOfImages))*100))

        idx += 1




