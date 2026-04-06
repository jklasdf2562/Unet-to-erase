import torch
import torch.nn as nn


class cnn_block(nn.Module):
    def __init__(self,in_chnnels,out_channels):
        super().__init__()
        self.conv1=nn.Conv2d(in_chnnels,out_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.act2=nn.ReLU(inplace=True)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
    def forward(self,x):
            x=self.conv1(x)
            x=self.bn1(x)
            x=self.act1(x)
            x=self.conv2(x)
            x=self.bn2(x)
            x=self.act2(x)
            skip=x
            x=self.pool(x)
            return x,skip


class encoder(nn.Module):
    def __init__(self,in_channels=3,embed_dim=1024): #[B,3,512,512]->[B,512,32,32]
        super().__init__()
        self.inchannels=in_channels
        self.channels=[(in_channels,64),(64,128),(128,256),(256,512)]
        self.blocks=nn.ModuleList([cnn_block(in_c,out_c) for in_c,out_c in self.channels])
        self.final_conv=nn.Conv2d(512,embed_dim,kernel_size=3,padding=1)
        self.act=nn.ReLU(inplace=True)
        self.final_conv2=nn.Conv2d(embed_dim,embed_dim,kernel_size=3,padding=1)
        self.act2=nn.ReLU(inplace=True)
    def forward(self,x):
        skips=[]
        for block in self.blocks:
            x,skip=block(x)
            skips.append(skip)
        x=self.final_conv(x)
        x=self.act(x)
        x=self.final_conv2(x)
        x=self.act2(x)
        return x,skips


class upblock(nn.Module):
    def __init__(self,in_channels,out_channels,skip_channels):
        super().__init__()
        self.upsample=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.skip_channels=skip_channels
        self.conv_pre=nn.Conv2d(in_channels,in_channels//2,kernel_size=1)
        self.conv1=nn.Conv2d(in_channels//2+skip_channels,out_channels,kernel_size=3,padding=1)
        self.bn1=nn.BatchNorm2d(out_channels)
        self.act1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        self.bn2=nn.BatchNorm2d(out_channels)
        self.act2=nn.ReLU(inplace=True)
    def forward(self,x,skip):
        x=self.upsample(x)
        x=self.conv_pre(x)
        x=torch.cat([x,skip],dim=1)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)      
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        return x


class decoder(nn.Module):
    def __init__(self,embed_dim=1024,out_channels=3,final_size=512): #[B,1024,32,32]->[B,3,512,512]
        super().__init__()
        self.final_size=final_size
        self.channels=[(embed_dim,512,512),(512,256,256),(256,128,128),(128,64,64)]
        self.skip_channels = [512,256,128,64]  
        self.upblocks=nn.ModuleList([
            upblock(in_c, out_c, skip_c) for (in_c, out_c, skip_c) in zip([embed_dim,512,256,128],[512,256,128,64],self.skip_channels)
        ])
        self.final_conv=nn.Conv2d(64,out_channels,kernel_size=3,padding=1)
        self.final_act=nn.Sigmoid()
    def forward(self,x,skips):
        for upblock, skip in zip(self.upblocks, skips[::-1]):#skips[64,128,256,512]
            x = upblock(x, skip)
        x = self.final_conv(x)
        x = self.final_act(x)

        return x


class Unet(nn.Module):
    def __init__(self,image_size=512,in_channels=3,embed_dim=512,out_channels=3):
        super().__init__()
        self.encoder=encoder(in_channels,embed_dim)
        self.decoder=decoder(embed_dim,out_channels,image_size)
    def forward(self,x):
        x,skips=self.encoder(x)
        x=self.decoder(x,skips)

        return x
