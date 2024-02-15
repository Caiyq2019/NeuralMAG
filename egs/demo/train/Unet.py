import torch
import torch.nn as nn


# 定义降采样部分
class downsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(downsample, self).__init__()
        self.down = nn.Sequential(
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kr, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_out) if drop_out>0 else nn.Identity()
        )

    def forward(self, x):
        return self.down(x)



# 定义上采样部分
class upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kr, s, p, drop_out):
        super(upsample, self).__init__()
        self.up = nn.Sequential(
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kr, stride=s, padding=p),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(drop_out) if drop_out>0 else nn.Identity()
            )
        
    def forward(self, x):
        
        return self.up(x)



# 32*32
class UNet(nn.Module):
    def __init__(self, kc=16, inc=3, ouc=3):
        super(UNet, self).__init__()

        self.pool = nn.AvgPool2d(2,2)
        self.s1 = nn.Conv2d(  inc,   1*kc,  3,1,1)  
        self.s2 = downsample( kc,    2*kc,  3,1,1, drop_out=0.0)  
        self.s3 = downsample( 2*kc,  4*kc,  3,1,1, drop_out=0.0)  
        self.s4 = downsample( 4*kc,  8*kc,  3,1,1, drop_out=0.0)  
        self.s5 = downsample( 8*kc,  16*kc, 3,1,1, drop_out=0.0)  

        self.up_1 = upsample( 16*kc, 8*kc,  4,2,1, drop_out=0.0)  
        self.up_2 = upsample( 16*kc, 4*kc,  4,2,1, drop_out=0.2)  
        self.up_3 = upsample( 8*kc,  2*kc,  4,2,1, drop_out=0.2)  
        self.up_4 = upsample( 4*kc,  1*kc,  4,2,1, drop_out=0.0)  
        

        self.last_Conv = nn.Sequential(
            nn.ConvTranspose2d(2*kc, 2*kc, 3,1,1),
            nn.BatchNorm2d(2*kc),
            nn.Tanh(),
            nn.Conv2d(2*kc, ouc, 1,1,0),
        )


        self.init_weight()
        self.out_channels = kc


    def init_weight(self):
        for w in self.modules():
            #判断层并且传参
            if isinstance(w, nn.Conv2d):
                #权重初始化
                nn.init.kaiming_normal_(w.weight, mode='fan_out')
                if w.bias is not None:
                    nn.init.zeros_(w.bias)
            elif isinstance(w, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(w.weight, mode='fan_in')
            elif isinstance(w, nn.BatchNorm2d):
                nn.init.ones_(w.weight)
                nn.init.zeros_(w.bias)
    



    def forward(self, x):   
        # down
        s1 = self.s1(x)
        down_1 = self.pool(s1)

        s2 = self.s2(down_1)
        down_2 = self.pool(s2)

        s3 = self.s3(down_2)
        down_3 = self.pool(s3)

        s4 = self.s4(down_3)
        down_4 = self.pool(s4)

        s5 = self.s5(down_4)

        # up
        up_1 = self.up_1(s5)
        up_2 = self.up_2(torch.cat([up_1, s4], dim=1))
        up_3 = self.up_3(torch.cat([up_2, s3], dim=1))
        up_4 = self.up_4(torch.cat([up_3, s2], dim=1))
        out  = self.last_Conv(torch.cat([up_4, s1], dim=1))

        return out
