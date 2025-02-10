import torch
import torch.nn as nn
import torch.nn.functional as F


class tg_adapter(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, length):
        super(tg_adapter, self).__init__()

        self.outer_param = nn.Parameter(torch.ones((length)))
        self.linear = nn.Linear(256, length)
        self.act = nn.ReLU(inplace=True)
        self.length = length
        # self.conv = nn.Conv2d()
        

    def forward(self, x):
        x = self.linear(x) # 128
        # print(self.outer_param.device)
        x = torch.outer(x.squeeze(0), self.outer_param)
        x = x.view(1,1,self.length,self.length)
        # print(x.shape)
        x = self.act(x)
        
        return x


class tg_adapter2(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, length):
        super(tg_adapter2, self).__init__()

        self.outer_param = nn.Parameter(torch.randn((length)))
        self.linear1 = nn.Linear(256, 128)
        self.linear2 = nn.Linear(128, length)
        self.act0 = nn.ReLU(inplace=True)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.length = length
        # self.conv = nn.Conv2d()
        

    def forward(self, x):
        x = self.linear1(x) # 128
        x = self.act0(x)
        x = self.linear2(x)
        x = self.act1(x)
        x = torch.outer(x.squeeze(0), self.outer_param)
        x = x.view(1,1,self.length,self.length)
        # print(x.shape)
        x = self.act2(x)
        
        return x

class down_sampler(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, factor):
        super(down_sampler, self).__init__()

        self.down_sampler_ = nn.Conv2d(3, 3, kernel_size=8, padding=1, stride=factor)
        
        # self.conv = nn.Conv2d()
        

    def forward(self, x):
    
        return self.down_sampler_(x)


import numpy as np
import torch
import torch.nn as nn 

class Downsampler(nn.Module):
    '''
        http://www.realitypixels.com/turk/computergraphics/ResamplingFilters.pdf
    '''
    def __init__(self, n_planes, factor, kernel_type, phase=0, kernel_width=None, support=None, sigma=None, preserve_size=False):
        super(Downsampler, self).__init__()
        
        assert phase in [0, 0.5], 'phase should be 0 or 0.5'

        if kernel_type == 'lanczos2':
            support = 2
            kernel_width = 4 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'lanczos3':
            support = 3
            kernel_width = 6 * factor + 1
            kernel_type_ = 'lanczos'

        elif kernel_type == 'gauss12':
            kernel_width = 7
            sigma = 1/2
            kernel_type_ = 'gauss'

        elif kernel_type == 'gauss1sq2':
            kernel_width = 9
            sigma = 1./np.sqrt(2)
            kernel_type_ = 'gauss'

        elif kernel_type in ['lanczos', 'gauss', 'box']:
            kernel_type_ = kernel_type

        else:
            assert False, 'wrong name kernel'
            
            
        # note that `kernel width` will be different to actual size for phase = 1/2
        self.kernel = get_kernel(factor, kernel_type_, phase, kernel_width, support=support, sigma=sigma)
        
        downsampler = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape, stride=factor, padding=0)
        downsampler.weight.data[:] = 0
        downsampler.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)
        for i in range(n_planes):
            downsampler.weight.data[i, i] = kernel_torch       

        self.downsampler_ = downsampler

        if preserve_size:

            if  self.kernel.shape[0] % 2 == 1: 
                pad = int((self.kernel.shape[0] - 1) / 2.)
            else:
                pad = int((self.kernel.shape[0] - factor) / 2.)
                
            self.padding = nn.ReplicationPad2d(pad)
        
        self.preserve_size = preserve_size
        
    def forward(self, input):
        if self.preserve_size:
            x = self.padding(input)
        else:
            x= input
        self.x = x
        return self.downsampler_(x)
        
def get_kernel(factor, kernel_type, phase, kernel_width, support=None, sigma=None):
    assert kernel_type in ['lanczos', 'gauss', 'box']
    
    # factor  = float(factor)
    if phase == 0.5 and kernel_type != 'box': 
        kernel = np.zeros([kernel_width - 1, kernel_width - 1])
    else:
        kernel = np.zeros([kernel_width, kernel_width])
    
        
    if kernel_type == 'box':
        assert phase == 0.5, 'Box filter is always half-phased'
        kernel[:] = 1./(kernel_width * kernel_width)
        
    elif kernel_type == 'gauss': 
        assert sigma, 'sigma is not specified'
        assert phase != 0.5, 'phase 1/2 for gauss not implemented'
        
        center = (kernel_width + 1.)/2.
        print(center, kernel_width)
        sigma_sq =  sigma * sigma
        
        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)
    elif kernel_type == 'lanczos': 
        assert support, 'support is not specified'
        center = (kernel_width + 1) / 2.

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                
                if phase == 0.5:
                    di = abs(i + 0.5 - center) / factor  
                    dj = abs(j + 0.5 - center) / factor 
                else:
                    di = abs(i - center) / factor
                    dj = abs(j - center) / factor
                
                
                pi_sq = np.pi * np.pi

                val = 1
                if di != 0:
                    val = val * support * np.sin(np.pi * di) * np.sin(np.pi * di / support)
                    val = val / (np.pi * np.pi * di * di)
                
                if dj != 0:
                    val = val * support * np.sin(np.pi * dj) * np.sin(np.pi * dj / support)
                    val = val / (np.pi * np.pi * dj * dj)
                
                kernel[i - 1][j - 1] = val
            
        
    else:
        assert False, 'wrong method name'
    
    kernel /= kernel.sum()
    
    return kernel


# class tg_adapter(nn.Module):
#     '''(conv => BN => ReLU) * 2'''
#     def __init__(self, length):
#         super(tg_adapter, self).__init__()

#         self.outer_param = nn.Parameter(torch.ones((length)))
#         self.linear1 = nn.Linear(256, 128)
#         self.linear2 = nn.Linear(128, length)
#         self.act = nn.ReLU(inplace=True)
#         self.length = length
#         # self.conv = nn.Conv2d()
        

#     def forward(self, x):
#         x = self.linear1(x) # 128
#         x = self.linear2(x)
#         # print(self.outer_param.device)
#         x = torch.outer(x.squeeze(0), self.outer_param)
#         x = x.view(1,1,self.length,self.length)
#         # print(x.shape)
#         x = self.act(x)
        
#         return x


# class up(nn.Module):
#     def __init__(self, in_ch, out_ch, bilinear=True):
#         super(up, self).__init__()

#         #  would be a nice idea if the upsampling could be learned too,
#         #  but my machine do not have enough memory to handle all those weights
#         if bilinear:
#             self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         else:
#             self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

#         self.conv = one_conv(in_ch, out_ch)

#     def forward(self, x1, x2, et):
#         x1 = self.up(x1 + et)
        
#         # input is CHW
#         diffY = x2.size()[2] - x1.size()[2]
#         diffX = x2.size()[3] - x1.size()[3]

#         x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                         diffY // 2, diffY - diffY//2))
        
#         # for padding issues, see 
#         # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#         # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

#         x = torch.cat([x2, x1], dim=1)
#         x = self.conv(x)
#         return x

