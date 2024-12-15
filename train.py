import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn as nn

from dataset import *
from testNet_5 import *
from utils import *


cuda_no = 0
dataset_file_path = '/data3/ALLPatches_Vid_Game.h5'
num_workers = 4
batch_size = 32
num_epoch = 30

net = TestNet_5(3, 16, True).cuda(cuda_no)
net.load_state_dict(torch.load("./TestNet5_Models/TestNet5_DIV2K_ep_30.pt"))
net.train()





criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)


train_dataset = trainDataSet(dataset_file_path)
train_loader = DataLoader(dataset = train_dataset, shuffle=True, num_workers=num_workers, batch_size=batch_size, pin_memory=False)


def Var(x):
    return Variable(x.cuda(cuda_no))

idx = 0

for epoch in range(1, num_epoch+1):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        
        hr, lr = Var(data[0]), Var(data[1])
        optimizer.zero_grad()


        outputs = net(lr)

        outputs = outputs.clamp(0., 1.)


        
        loss = criterion(outputs, hr)
        running_loss += loss.detach().item()

        loss.backward()
        optimizer.step()


        # print statistics
                
        if i % batch_size == 0:
            print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / batch_size}')
            running_loss = 0.0
    torch.save(net.state_dict(), './TestNet5_Models/TestNet5_VidGame_ep_' + str(epoch) + '.pt')

print('Finished Training')







