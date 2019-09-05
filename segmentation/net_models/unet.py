'''
https://github.com/kevinlu1211
'''
import torch
import torch.nn as nn
import torchvision

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, n_classes=2):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=True)
        down_blocks = []
        up_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
                                                    up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.out = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1, stride=1),
            nn.LogSoftmax()
        )
        
        # --------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )

    def forward(self, x, with_output_feature_map=False):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (UNetWithResnet50Encoder.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        class_x = x
        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])
        
        x = self.out(x)
        aux = nn.functional.adaptive_max_pool2d(input=class_x, output_size=(1, 1)).view(-1, class_x.size(1))
        
#        del pre_pools
#        del class_x
#        
        return x, self.classifier(aux)

#model = UNetWithResnet50Encoder().cuda()
#inp = torch.rand((2, 3, 256, 512)).cuda()
#out, out_cls = model(inp)

'''
Code obtained from repository: https://github.com/milesial/Pytorch-UNet
'''

#
#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#
#class inconv(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(inconv, self).__init__()
#        self.conv = double_conv(in_ch, out_ch)
#
#    def forward(self, x):
#        x = self.conv(x)
#        return x
#
#
#class down(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(down, self).__init__()
#        self.mpconv = nn.Sequential(
#            nn.MaxPool2d(2),
#            double_conv(in_ch, out_ch)
#        )
#
#    def forward(self, x):
#        x = self.mpconv(x)
#        return x
#
#
#class up(nn.Module):
#    def __init__(self, in_ch, out_ch, bilinear=True):
#        super(up, self).__init__()
#
#        #  would be a nice idea if the upsampling could be learned too,
#        #  but my machine do not have enough memory to handle all those weights
#        if bilinear:
#            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#        else:
#            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)
#
#        self.conv = double_conv(in_ch, out_ch)
#
#    def forward(self, x1, x2):
#        x1 = self.up(x1)
#        
#        # input is CHW
#        diffY = x2.size()[2] - x1.size()[2]
#        diffX = x2.size()[3] - x1.size()[3]
#
#        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
#                        diffY // 2, diffY - diffY//2))
#        
#        # for padding issues, see 
#        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
#        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
#
#        x = torch.cat([x2, x1], dim=1)
#        x = self.conv(x)
#        return x
#
#
#class outconv(nn.Module):
#    def __init__(self, in_ch, out_ch):
#        super(outconv, self).__init__()
#        self.conv = nn.Conv2d(in_ch, out_ch, 1)
#
#    def forward(self, x):
#        x = self.conv(x)
#        return x
#    
#class UNet(nn.Module):
#    def __init__(self, n_channels, n_classes):
#        super(UNet, self).__init__()
#        self.inc = inconv(n_channels, 64)
#        self.down1 = down(64, 128)
#        self.down2 = down(128, 256)
#        self.down3 = down(256, 512)
#        self.down4 = down(512, 512)
#        self.up1 = up(1024, 256)
#        self.up2 = up(512, 128)
#        self.up3 = up(256, 64)
#        self.up4 = up(128, 64)
#        self.outc = outconv(64, n_classes)
#
#    def forward(self, x):
#        x1 = self.inc(x)
#        x2 = self.down1(x1)
#        x3 = self.down2(x2)
#        x4 = self.down3(x3)
#        x5 = self.down4(x4)
#        x = self.up1(x5, x4)
#        x = self.up2(x, x3)
#        x = self.up3(x, x2)
#        x = self.up4(x, x1)
#        x = self.outc(x)
#        return F.sigmoid(x)
#
#class double_conv(nn.Module):
#    '''(conv => BN => ReLU) * 2'''
#    def __init__(self, in_ch, out_ch):
#        super(double_conv, self).__init__()
#        self.conv = nn.Sequential(
#            nn.Conv2d(in_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True),
#            nn.Conv2d(out_ch, out_ch, 3, padding=1),
#            nn.BatchNorm2d(out_ch),
#            nn.ReLU(inplace=True)
#        )
#
#    def forward(self, x):
#        x = self.conv(x)
#        return x
