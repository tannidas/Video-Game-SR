import torch.nn as nn
import torch.nn.functional as F
import torch


def conv2D(in_channels, out_channels, kernel_size=3):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding = (kernel_size//2))

def activation():
    return nn.PReLU()

def maxPooling2D(kernel_size=2):
    return nn.MaxPool2d(kernel_size=kernel_size, stride=kernel_size)



class basic_block(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, is_max_pool=True):
        super(basic_block, self).__init__()
        self.conv = conv2D(in_channels, out_channels, kernel_size=kernel_size)
        self.activ = activation()
        self.is_max_pool = is_max_pool
        self.max_pool = maxPooling2D()
        self.dropout = nn.Dropout(p=0.25)
        # self.pixel_shuffle = nn.PixelShuffle(1)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        if self.is_max_pool:
            x = self.max_pool(x)
        return x

class tail_block(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(tail_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1, padding=0)
        self.activ = nn.Tanh()
        self.dropout = nn.Dropout(p=0.25)
        torch.nn.init.xavier_uniform_(self.conv.weight)

    def forward(self, x):
        x = self.conv(x)
        x = self.activ(x)
        return x


class TestNet_5(nn.Module):

    def __init__(self, in_channels, n_basic_blocks, is_with_res):
        super(TestNet_5, self).__init__()
        
        self.net_head = basic_block(in_channels, 128, 1, False)

        net_body = [basic_block(128, 128, 3, False) for _ in range(n_basic_blocks)]
        self.net_body = nn.Sequential(*net_body)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.net_tail = tail_block(128, 48)

        self.is_with_res = is_with_res

    def forward(self, x):

        first_layer = self.net_head(x)

        mid_layers = self.net_body(first_layer)

        interim_output = self.net_tail(mid_layers)
        
        pixel_shuffle_1 = self.pixel_shuffle(interim_output)
        pixel_shuffle_2 = self.pixel_shuffle(pixel_shuffle_1)

        # Residual Connection
        if self.is_with_res:
            final_output = pixel_shuffle_2 + F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        else:
            final_output = pixel_shuffle_2
        

        return final_output

if __name__=='__main__':
    input = torch.zeros(1, 3, 64, 64)
    model = TestNet_5(3, 16, True)
    output = model(input)
    print("output_size: ", output.size())