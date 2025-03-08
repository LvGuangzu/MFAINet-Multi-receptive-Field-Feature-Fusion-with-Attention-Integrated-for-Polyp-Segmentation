import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.pool_pvt_3 import pvt_v2_b3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGlobalConv(nn.Module):
    def __init__(self, in_channels):
        super(LocalGlobalConv, self).__init__()
        # Local convolutions
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        # Global convolution
        self.conv7x7 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)
        # Fusion layer for local features
        self.local_fusion = nn.Conv2d(in_channels * 3, in_channels, kernel_size=1)

    def forward(self, x):
        # Local features
        local1 = self.conv1x1(x)
        local3 = self.conv3x3(x)
        local5 = self.conv5x5(x)
        # Concatenate local features
        local_combined = torch.cat([local1, local3, local5], dim=1)
        # Fuse local features
        local_fused = self.local_fusion(local_combined)
        # Global features
        global_fused = self.conv7x7(x)
        return local_fused, global_fused


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out) * x
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        x = self.sigmoid(x) * x
        return x


class Attention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(Attention, self).__init__()
        self.channelAttention = ChannelAttention(in_planes)
        self.spatialAttention = SpatialAttention()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        add_x_y = x + y
        channelAttention = self.channelAttention(add_x_y)
        spatialAttention = self.spatialAttention(add_x_y)
        add_attention = channelAttention + spatialAttention
        sigmoid_add_attention = self.sigmoid(add_attention)
        product_x = sigmoid_add_attention * x
        product_y = sigmoid_add_attention * y
        add_product = product_x + product_y
        return add_product

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out, factor):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=factor),
            nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x
    
# 模型开始
class MFAINet(nn.Module):
    def __init__(self, channel=32):
        super(MFAINet, self).__init__()

        self.backbone = pvt_v2_b3()  # [64, 128, 320, 512]
        path = './pretrained_pth/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        # 通过预训练模型的参数初始化骨架网络
        self.backbone.load_state_dict(model_dict)
        
        self.LocalGlobalConv1 = LocalGlobalConv(64)
        self.LocalGlobalConv2 = LocalGlobalConv(128)
        self.LocalGlobalConv3 = LocalGlobalConv(320)
        self.LocalGlobalConv4 = LocalGlobalConv(512)
        
        self.Attention1 = Attention(64)
        self.Attention2 = Attention(128)
        self.Attention3 = Attention(320)
        self.Attention4 = Attention(512)
        
        self.up_conv1 = up_conv(512, 320, 2);
        self.up_conv2 = up_conv(320, 128, 2);
        self.up_conv3 = up_conv(128, 64, 2);
        
        self.up_conv4 = up_conv(512, 64, 8);
        self.up_conv5 = up_conv(320, 64, 4);
        self.up_conv6 = up_conv(128, 64, 2);
        
        self.up_conv7 = up_conv(512, 320, 2)
        self.up_conv8 = up_conv(512, 128, 4)
        self.up_conv9 = up_conv(512, 64, 8)
        
        self.up_conv10 = up_conv(320, 128, 2)
        self.up_conv11 = up_conv(320, 64, 4)
        
        self.up_conv12 = up_conv(128, 64, 2)
        
        self.beforelinear = nn.Conv2d(64*4, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=True)
        
        self.headconv = nn.Sequential(
                            nn.Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=True),
                            nn.Conv2d(32, 1, kernel_size=(1, 1), stride=(1, 1), bias=True)
                        )
        
    def forward(self, x):
        # backbone pvt
        pvt = self.backbone(x)

        x1 = pvt[0]  # torch.Size([1, 64, 88, 88])
        x2 = pvt[1]  # torch.Size([1, 128, 44, 44])
        x3 = pvt[2]  # torch.Size([1, 320, 22, 22])
        x4 = pvt[3]  # torch.Size([1, 512, 11, 11])
        
        up_conv_output1 = self.up_conv7(x4)
        up_conv_output2 = self.up_conv8(x4)
        up_conv_output3 = self.up_conv9(x4)
        
        up_conv_output4 = self.up_conv10(x3)
        up_conv_output5 = self.up_conv11(x3)
        
        up_conv_output6 = self.up_conv12(x2)
        
        
        product1 = x3 * up_conv_output1 # torch.Size([1, 320, 22, 22])
        product2 = x2 * up_conv_output2 * up_conv_output4 # torch.Size([1, 128, 44, 44])
        product3 = x1 * up_conv_output3 * up_conv_output5 * up_conv_output6
        
        LocalGlobalConv1_output1, LocalGlobalConv1_output2 = self.LocalGlobalConv1(product3) # torch.Size([1, 64, 88, 88])
        Attention1_output = self.Attention1(LocalGlobalConv1_output1, LocalGlobalConv1_output2) # torch.Size([1, 64, 88, 88])
        
        LocalGlobalConv2_output1, LocalGlobalConv2_output2 = self.LocalGlobalConv2(product2) # torch.Size([1, 128, 44, 44])
        Attention2_output = self.Attention2(LocalGlobalConv2_output1, LocalGlobalConv2_output2) # torch.Size([1, 128, 44, 44])
        
        LocalGlobalConv3_output1, LocalGlobalConv3_output2 = self.LocalGlobalConv3(product1) # torch.Size([1, 320, 22, 22])
        Attention3_output = self.Attention3(LocalGlobalConv3_output1, LocalGlobalConv3_output2) # torch.Size([1, 320, 22, 22])
        
        LocalGlobalConv4_output1, LocalGlobalConv4_output2 = self.LocalGlobalConv4(x4) # torch.Size([1, 512, 11, 11])
        Attention4_output = self.Attention4(LocalGlobalConv4_output1, LocalGlobalConv4_output2) # torch.Size([1, 512, 11, 11])
        
        
        #Attention1_output torch.Size([1, 64, 88, 88])
        #Attention2_output torch.Size([1, 128, 44, 44])
        #Attention3_output torch.Size([1, 320, 22, 22])
        #Attention4_output torch.Size([1, 512, 11, 11])
 
        up_conv1_output = self.up_conv1(Attention4_output) # torch.Size([1, 320, 22, 22])
        up_conv2_output = self.up_conv2(Attention3_output+up_conv1_output) # torch.Size([1, 128, 44, 44])
        up_conv3_output = self.up_conv3(Attention2_output+up_conv2_output) # torch.Size([1, 64, 88, 88])
        
        up_conv4_output = self.up_conv4(Attention4_output) # torch.Size([1, 64, 88, 88])
        up_conv5_output = self.up_conv5(Attention3_output) # torch.Size([1, 64, 88, 88])
        up_conv6_output = self.up_conv6(Attention2_output) # torch.Size([1, 64, 88, 88])
        
        Cat_output = torch.cat([up_conv4_output, up_conv5_output, up_conv6_output, Attention1_output], 1)
        
        beforelinear = self.beforelinear(Cat_output)
        
        prediction1 = self.headconv(up_conv3_output)
        prediction2 = self.headconv(beforelinear)
        
        prediction1 = F.interpolate(prediction1, scale_factor=4, mode='bilinear')
        prediction2 = F.interpolate(prediction2, scale_factor=4, mode='bilinear')
        
        return prediction1, prediction2


if __name__ == '__main__':
    model = MFAINet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    
    model(input_tensor)
    prediction1, prediction2 = model(input_tensor)
    print(prediction1.size(), prediction2.size())

