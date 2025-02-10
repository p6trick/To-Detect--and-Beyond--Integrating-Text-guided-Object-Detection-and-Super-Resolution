import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch.nn as nn


from .modules import *
from .TG_modules import *


# class TG_UNet2(nn.Module):
#     def __init__(self, n_channels, n_classes, input_shape):
#         super(TG_UNet2, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128) # 256 -> 128
#         self.down2 = down(128, 256) # 128 -> 64
#         self.down3 = down(256, 512) # 64 -> 32
#         self.down4 = down(512, 512) # 32 -> 16
#         # self.down_layers = [down1, down2, down3, down4]

#         self.up1 = up(1024, 256,bilinear=False)#,bilinear=False 16 -> 32
#         self.up2 = up(512, 128,bilinear=False) # 32 -> 64
#         self.up3 = up(256, 64,bilinear=False) # 64 -> 128
#         self.up4 = up(128, 32,bilinear=False)# 128 -> 256
#         self.up5 = up(64, 32,bilinear=False) # 256 -> 512
#         # self.up_layers = [up1, up2, up3, up4, up5]

#         self.outc = outconv(32, n_classes)#64

#         self.up_s1=up_s(64,32)

#         #########################################################
#         self.h, self.w = input_shape
#         # self.down_txt_layers = [tg_adapter(self.h//2**i) for i in range(len(self.down_layers))]

#         # self.up_txt_layers = [tg_adapter((self.h//2**len(self.down_layers))*i) for i in range(len(self.up_layers))]
#         self.dwn_txt1 = tg_adapter2(256)
#         self.dwn_txt2 = tg_adapter2(128)
#         self.dwn_txt3 = tg_adapter2(64)
#         self.dwn_txt4 = tg_adapter2(32)
        
#         #self.up_s=up_s(32,16)
#         #self.up_s=up_s(16,8)

#     def forward(self, x, encoded_text):
        
#         et = encoded_text[:,0,:]
        
#         ############################################
#         # for dl,dtl in zip(self.down_layers, self.down_txt_layers):
            
#         x1 = self.inc(x)
#         # t1 = self.down_txt_layers[0](et)
#         # print(x1.shape)
#         t1 = self.dwn_txt1(et)

#         x2 = self.down1(x1 + t1)
#         # t2 = self.down_txt_layers[1](et)
#         t2 = self.dwn_txt2(et)
#         # print(x2.shape, t2.shape)
#         x3 = self.down2(x2 + t2)
#         # t3 = self.down_txt_layers[2](et)
#         t3 = self.dwn_txt3(et)
        
#         x4 = self.down3(x3 + t3)
#         # t4 = self.down_txt_layers[3](et)
#         t4 = self.dwn_txt4(et)

#         x5 = self.down4(x4 + t4)

#         x = self.up1(x5, x4)
#         # print(f'x: {x.shape}')
#         x = self.up2(x, x3)
#         #print(x.shape)
#         x = self.up3(x, x2)
#         #print(x.shape)
#         x = self.up4(x, x1)
#         #print(x.shape)
#         #print("x0")
#         x0 = self.up_s1(x1)
#         #print(x0.shape)
#         x = self.up5(x, x0)

#         #xout2 = F.conv2d(x2.unsqueeze(1), self.weight1, padding=2)
#         #xout3 = F.conv2d(x3.unsqueeze(1), self.weight1, padding=2)
#         #x = self.beforeconv(x)
#         #x = self.pixel_shuffle(x)
#         x = self.outc(x)
#         return x#torch.sigmoid(x)

#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)


# def normal_init(m, mean, std):
#     if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
#         m.weight.data.normal_(mean, std)
#         m.bias.data.zero_()

# def imshow(img):
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#     plt.pause(5)

# class TG_UNet4(nn.Module):
#     def __init__(self, n_channels, n_classes, input_shape):
#         super(TG_UNet4, self).__init__()
#         self.inc = inconv(n_channels, 64)
#         self.down1 = down(64, 128)
#         self.down2 = down(128, 256)
#         self.down3 = down(256, 512)
#         self.down4 = down(512, 512)
#         self.up1 = up(1024, 256,bilinear=False)
#         self.up2 = up(512, 128,bilinear=False)
#         self.up3 = up(256, 64,bilinear=False)
#         self.up4 = up(128, 32,bilinear=False)#(128, 64)
#         self.up5 = up(64, 16,bilinear=False)
#         self.up6 = up(32, 16,bilinear=False)
#         self.outc = outconv(16, n_classes)#64

#         self.up_s1=up_s(64,32)
#         self.up_s2=up_s(32,16)
#         #self.up_s=up_s(16,8)

#         self.h, self.w = input_shape
#         # self.down_txt_layers = [tg_adapter(self.h//2**i) for i in range(len(self.down_layers))]

#         # self.up_txt_layers = [tg_adapter((self.h//2**len(self.down_layers))*i) for i in range(len(self.up_layers))]
#         self.dwn_txt1 = tg_adapter2(128)
#         self.dwn_txt2 = tg_adapter2(64)
#         self.dwn_txt3 = tg_adapter2(32)
#         self.dwn_txt4 = tg_adapter2(16)

        
#         self.up_txt0 = tg_adapter(256)

#     def forward(self, x, encoded_text):

#         et = encoded_text[:,0,:]
#         x1 = self.inc(x)
#         # t1 = self.down_txt_layers[0](et)
#         # print(x1.shape)
#         t1 = self.dwn_txt1(et)

#         x2 = self.down1(x1 + t1)
#         # t2 = self.down_txt_layers[1](et)
#         t2 = self.dwn_txt2(et)
#         # print(x2.shape, t2.shape)
#         x3 = self.down2(x2 + t2)
#         # t3 = self.down_txt_layers[2](et)
#         t3 = self.dwn_txt3(et)
        
#         x4 = self.down3(x3 + t3)
#         # t4 = self.down_txt_layers[3](et)
#         t4 = self.dwn_txt4(et)
        
#         x5 = self.down4(x4 + t4)

#         x = self.up1(x5, x4)
#         #print(x.shape)
#         x = self.up2(x, x3)
#         #print(x.shape)
#         x = self.up3(x, x2)
#         #print(x.shape)
#         x = self.up4(x, x1)
#         #print(x.shape)
#         #print("x0")
        
#         x0=self.up_s1(x1)
#         ut0 = self.up_txt0(et)

#         x_1=self.up_s2(x0+ut0)
#         #print(x0.shape)
#         x = self.up5(x, x0)
#         x = self.up6(x, x_1)
#         #print("up5")
#         #print(x.shape)
#         x = self.outc(x)
#         #print("outc")
#         #print(x.shape)


#         return torch.sigmoid(x)

#     def weight_init(self, mean, std):
#         for m in self._modules:
#             normal_init(self._modules[m], mean, std)


class TG_UNet2x2(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape):
        super(TG_UNet2x2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128) # 256 -> 128
        self.down2 = down(128, 256) # 128 -> 64
        self.down3 = down(256, 512) # 64 -> 32
        self.down4 = down(512, 512) # 32 -> 16
        # self.down_layers = [down1, down2, down3, down4]

        self.up1 = up(1024, 256,bilinear=False)#,bilinear=False 16 -> 32
        self.up2 = up(512, 128,bilinear=False) # 32 -> 64
        self.up3 = up(256, 64,bilinear=False) # 64 -> 128
        self.up4 = up(128, 32,bilinear=False)# 128 -> 256
        self.up5 = up(64, 32,bilinear=False) # 256 -> 512
        # self.up_layers = [up1, up2, up3, up4, up5]

        self.outc = outconv(32, n_classes)#64

        self.up_s1=up_s(64,32)

        #########################################################
        self.h, self.w = input_shape
        # self.down_txt_layers = [tg_adapter(self.h//2**i) for i in range(len(self.down_layers))]

        # self.up_txt_layers = [tg_adapter((self.h//2**len(self.down_layers))*i) for i in range(len(self.up_layers))]
        self.dwn_txt1 = tg_adapter2(256)
        self.dwn_txt2 = tg_adapter2(128)
        self.dwn_txt3 = tg_adapter2(64)
        self.dwn_txt4 = tg_adapter2(32)
        
        
        #self.up_s=up_s(32,16)
        #self.up_s=up_s(16,8)

    def forward(self, x, encoded_text):
        
        et = encoded_text[:,0,:]
        
        ############################################
        # for dl,dtl in zip(self.down_layers, self.down_txt_layers):
            
        x1 = self.inc(x)
        # t1 = self.down_txt_layers[0](et)
        # print(x1.shape)
        t1 = self.dwn_txt1(et)

        x2 = self.down1(x1 + t1)
        # t2 = self.down_txt_layers[1](et)
        t2 = self.dwn_txt2(et)
        # print(x2.shape, t2.shape)
        x3 = self.down2(x2 + t2)
        # t3 = self.down_txt_layers[2](et)
        t3 = self.dwn_txt3(et)
        
        x4 = self.down3(x3 + t3)
        # t4 = self.down_txt_layers[3](et)
        t4 = self.dwn_txt4(et)

        x5 = self.down4(x4 + t4)

        x = self.up1(x5, x4)
        # print(f'x: {x.shape}')
        x = self.up2(x, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        #print("x0")
        x0 = self.up_s1(x1)
        #print(x0.shape)
        x = self.up5(x, x0)

        #xout2 = F.conv2d(x2.unsqueeze(1), self.weight1, padding=2)
        #xout3 = F.conv2d(x3.unsqueeze(1), self.weight1, padding=2)
        #x = self.beforeconv(x)
        #x = self.pixel_shuffle(x)
        x = self.outc(x)
        return x#torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.show()
    plt.pause(5)

class TG_UNet4x2(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape):
        super(TG_UNet4x2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256,bilinear=False)
        self.up2 = up(512, 128,bilinear=False)
        self.up3 = up(256, 64,bilinear=False)
        self.up4 = up(128, 32,bilinear=False)#(128, 64)
        self.up5 = up(64, 16,bilinear=False)
        self.up6 = up(32, 16,bilinear=False)
        self.outc = outconv(16, n_classes)#64

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)
        #self.up_s=up_s(16,8)

        self.h, self.w = input_shape
        # self.down_txt_layers = [tg_adapter(self.h//2**i) for i in range(len(self.down_layers))]

        # self.up_txt_layers = [tg_adapter((self.h//2**len(self.down_layers))*i) for i in range(len(self.up_layers))]
        self.dwn_txt1 = tg_adapter2(128)
        self.dwn_txt2 = tg_adapter2(64)
        self.dwn_txt3 = tg_adapter2(32)
        self.dwn_txt4 = tg_adapter2(16)

        
        self.up_txt0 = tg_adapter2(256)

    def forward(self, x, encoded_text):

        et = encoded_text[:,0,:]
        x1 = self.inc(x)
        # t1 = self.down_txt_layers[0](et)
        # print(x1.shape)
        t1 = self.dwn_txt1(et)

        x2 = self.down1(x1 + t1)
        # t2 = self.down_txt_layers[1](et)
        t2 = self.dwn_txt2(et)
        # print(x2.shape, t2.shape)
        x3 = self.down2(x2 + t2)
        # t3 = self.down_txt_layers[2](et)
        t3 = self.dwn_txt3(et)
        
        x4 = self.down3(x3 + t3)
        # t4 = self.down_txt_layers[3](et)
        t4 = self.dwn_txt4(et)
        
        x5 = self.down4(x4 + t4)

        x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        #print("x0")
        
        x0=self.up_s1(x1)
        ut0 = self.up_txt0(et)

        x_1=self.up_s2(x0+ut0)
        #print(x0.shape)
        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        #print("up5")
        #print(x.shape)
        x = self.outc(x)
        #print("outc")
        #print(x.shape)


        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

class TG_UNet8x2(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape):
        super(TG_UNet8x2, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)#(128, 64)
        self.up5 = up(64, 16)
        self.up6 = up(32,8)
        self.up7 = up(16,8)
        self.outc = outconv(8, n_classes)#64

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)
        self.up_s3=up_s(16,8)

        self.h, self.w = input_shape
        # self.down_txt_layers = [tg_adapter(self.h//2**i) for i in range(len(self.down_layers))]

        # self.up_txt_layers = [tg_adapter((self.h//2**len(self.down_layers))*i) for i in range(len(self.up_layers))]
        self.dwn_txt1 = tg_adapter2(64)
        self.dwn_txt2 = tg_adapter2(32)
        self.dwn_txt3 = tg_adapter2(16)
        self.dwn_txt4 = tg_adapter2(8)

        self.up_txt0 = tg_adapter2(128)
        self.up_txt1 = tg_adapter2(256)
        

    def forward(self, x,encoded_text ):
        #print(x.shape)

        et = encoded_text[:,0,:]
        x1 = self.inc(x)

        t1 = self.dwn_txt1(et)
        # print(x1.shape, t1.shape)
        x2 = self.down1(x1+t1)
        t2 = self.dwn_txt2(et)
        #print(x2.shape)
        x3 = self.down2(x2+t2)
        #print(x3.shape)
        t3 = self.dwn_txt3(et)
        x4 = self.down3(x3+t3)
        #print(x4.shape)
        t4 = self.dwn_txt4(et)
        x5 = self.down4(x4+t4)
        #print("x5")
        #print(x5.shape)

        x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        #print("x0")
        x0=self.up_s1(x1)
        ut0 = self.up_txt0(et)
        x_1=self.up_s2(x0+ut0)
        ut1 = self.up_txt1(et)
        x_2=self.up_s3(x_1+ut1)
        #print(x0.shape)
        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.up7(x,x_2)
        #print("up5")
        #print(x.shape)
        x = self.outc(x)
        #print("outc")
        #print(x.shape)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)


class TG_UNet8x2_SF(nn.Module):
    def __init__(self, n_channels, n_classes, input_shape):
        super(TG_UNet8x2_SF, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 32)#(128, 64)
        self.up5 = up(64, 16)
        self.up6 = up(32,8)
        self.up7 = up(16,8)
        self.outc = outconv(8, n_classes)#64

        self.up_s1=up_s(64,32)
        self.up_s2=up_s(32,16)
        self.up_s3=up_s(16,8)

        self.h, self.w = input_shape
        # self.down_txt_layers = [tg_adapter(self.h//2**i) for i in range(len(self.down_layers))]

        # self.up_txt_layers = [tg_adapter((self.h//2**len(self.down_layers))*i) for i in range(len(self.up_layers))]
        self.dwn_txt1 = tg_adapter2(128)
        self.dwn_txt2 = tg_adapter2(64)
        self.dwn_txt3 = tg_adapter2(32)
        self.dwn_txt4 = tg_adapter2(16)

        self.up_txt0 = tg_adapter2(256)
        self.up_txt1 = tg_adapter2(512)
        
        self.sf = nn.Softmax(dim=-1)

        

    def forward(self, x,encoded_text ):
        #print(x.shape)

        et = encoded_text[:,0,:]
        x1 = self.inc(x)

        t1 = self.dwn_txt1(et)
        tt1 = t1@x1.mean(dim=1).transpose(-2,-1)
        # print(x1.shape, t1.shape)
        x2 = self.down1(x1+tt1)
        t2 = self.dwn_txt2(et)
        tt2 = t2@x2.mean(dim=1).transpose(-2,-1)
        #print(x2.shape)
        x3 = self.down2(x2+tt2)
        #print(x3.shape)
        t3 = self.dwn_txt3(et)
        tt3 = t3@x3.mean(dim=1).transpose(-2,-1)
        x4 = self.down3(x3+tt3)
        #print(x4.shape)
        t4 = self.dwn_txt4(et)
        tt4 = t4@x4.mean(dim=1).transpose(-2,-1)
        x5 = self.down4(x4+tt4)
        #print("x5")
        #print(x5.shape)

        x = self.up1(x5, x4)
        #print(x.shape)
        x = self.up2(x, x3)
        #print(x.shape)
        x = self.up3(x, x2)
        #print(x.shape)
        x = self.up4(x, x1)
        #print(x.shape)
        #print("x0")
        x0=self.up_s1(x1)
        ut0 = self.up_txt0(et)
        utt0 = ut0@x0.mean(dim=1).transpose(-2,-1)

        x_1=self.up_s2(x0+utt0)
        ut1 = self.up_txt1(et)
        
        utt1 = ut1@x_1.mean(dim=1).transpose(-2,-1)
        x_2=self.up_s3(x_1+utt1)
        #print(x0.shape)
        x = self.up5(x, x0)
        x = self.up6(x, x_1)
        x = self.up7(x,x_2)
        #print("up5")
        #print(x.shape)
        x = self.outc(x)
        #print("outc")
        #print(x.shape)
        return torch.sigmoid(x)

    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)