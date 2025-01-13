import torch.nn as nn
import torchvision.models as models
import torch



class AlexNet(nn.Module):
    def __init__(self, nums_classes):
        super().__init__()
        self.features = nn.Sequential(*list(models.alexnet(pretrained=True).features.children())[:-1])
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
        self.features = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])
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
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-3])
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
        self.features = nn.Sequential(*list(models.densenet169(pretrained=True).features.children())[:-3])
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
    else:
        raise ValueError("net_type must be [`alexnet-GAP`,`vgg16-GAP`,`resnet50-GAP`,`densenet169-GAP`]")
    return net







if __name__ == "__main__":
    net = DenseNet169(10)
    x = torch.randn(size=(32,3,224,224))
    y = net(x)
    print(y.shape)

