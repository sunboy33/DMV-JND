from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])

v_transform = transforms.Compose([transforms.Resize((224,224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])


def get_dataloader(batch_size):
    t_dataset = CIFAR10(root="datasets",train=True,transform=transform,download=True)
    v_dataset = CIFAR10(root="datasets",train=False,transform=transform,download=True)
    td = DataLoader(t_dataset,shuffle=True,batch_size=batch_size,num_workers=2)
    vd = DataLoader(v_dataset,shuffle=False,batch_size=batch_size,num_workers=2)
    return {"train":td,"val":vd}

