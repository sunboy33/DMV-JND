import torch.nn as nn
import torchvision.models as models
import torch


class AlexNet(nn.Module):
    def __init__(self, nums_classes):
        super().__init__()
        self.features = nn.Sequential(*list(models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).features.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256, nums_classes))
    
    def forward(self,x):
        x = self.features(x)
        x = x.mean(dim=(2,3))
        x = self.classifier(x)
        return x

class VGG16(nn.Module):
    def __init__(self, nums_classes):
        super().__init__()
        self.features = nn.Sequential(*list(models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features.children())[:-1])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, nums_classes))
    
    def forward(self,x):
        x = self.features(x)
        x = x.mean(dim=(2,3))
        x = self.classifier(x)
        return x

class ResNet50(nn.Module):
    def __init__(self, nums_classes):
        super().__init__()
        self.features = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).children())[:-3])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1024, nums_classes))
    
    def forward(self,x):
        x = self.features(x)
        x = x.mean(dim=(2,3))
        x = self.classifier(x)
        return x
    
class DenseNet169(nn.Module):
    def __init__(self, nums_classes):
        super().__init__()
        self.features = nn.Sequential(*list(models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1).features.children())[:-3])
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280, nums_classes))
    
    def forward(self,x):
        x = self.features(x)
        x = x.mean(dim=(2,3))
        x = self.classifier(x)
        return x

class MobileNet(nn.Module):
    def __init__(self, nums_classes):
        super().__init__()
        self.features = nn.Sequential(*list(models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1).features.children()))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(1280, nums_classes))
    
    def forward(self,x):
        x = self.features(x)
        x = x.mean(dim=(2,3))
        x = self.classifier(x)
        return x

def get_net(net_type,num_classes):
    if net_type=="alexnet-GAP":
        net = AlexNet(num_classes)
    elif net_type=="vgg16-GAP":
        net = VGG16(num_classes)
    elif net_type=="resnet50-GAP":
        net = ResNet50(num_classes)
    elif net_type=="densenet169-GAP":
        net = DenseNet169(num_classes)
    elif net_type=="mobilenet-GAP":
        net = MobileNet(num_classes)
    else:
        raise ValueError("net_type must be [`alexnet-GAP`,`vgg16-GAP`,`resnet50-GAP`,`densenet169-GAP`,`mobilenet-GAP`]")
    return net







if __name__ == "__main__":
    net = MobileNet(10)
    x = torch.randn(size=(32,3,224,224))
    y = net(x)
    print(y.shape)

