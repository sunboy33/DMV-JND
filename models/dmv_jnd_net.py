import torch.nn as nn
import torch



class JNDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(in_channels=4,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU())
        self.encoder2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),stride=1,padding=1),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU(),
                                      nn.MaxPool2d(kernel_size=(2,2)))
        self.encoder3 = nn.Sequential(nn.Conv2d(in_channels=128,out_channels=256,kernel_size=(3,3),stride=1,padding=1),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU(),
                                      nn.MaxPool2d(kernel_size=(2,2)))
        self.encoder4 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=64,kernel_size=(3,3),stride=1,padding=1),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU(),
                                      nn.MaxPool2d(kernel_size=(2,2)))

        self.decoder1 = nn.Sequential(nn.ConvTranspose2d(in_channels=64,out_channels=256,kernel_size=(2,2),stride=2),
                                      nn.BatchNorm2d(256),
                                      nn.LeakyReLU())
        self.decoder2 = nn.Sequential(nn.ConvTranspose2d(in_channels=256,out_channels=128,kernel_size=(2,2),stride=2),
                                      nn.BatchNorm2d(128),
                                      nn.LeakyReLU())
        self.decoder3 = nn.Sequential(nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(2,2),stride=2),
                                      nn.BatchNorm2d(64),
                                      nn.LeakyReLU())
        self.decoder4 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=3,kernel_size=(3,3),stride=1,padding=1),
                                      nn.Tanh())
        
    def forward(self,x,c):
        input = torch.cat([x,c],dim=1)
        x = self.encoder1(input)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        output = self.decoder4(x)
        return output

if __name__=="__main__":
    x = torch.randn(size=(32,3,32,32))
    c = torch.randn(size=(32,1,32,32))
    net = JNDNet()
    o = net(x,c)
    print(o.shape)

