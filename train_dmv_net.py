import os

import torch.nn.functional
from torchvision import transforms
import torch
from torchvision.utils import make_grid
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image

from models import JNDNet
from loss import Loss
from datasets import get_dataloader



def get_cam(x,classifier_nets,device):
    x = transforms.Resize((224,224))(x)
    heats = torch.zeros(size=(x.shape[0],1,32,32),device=device)
    for net_type in classifier_nets.keys():
        net = classifier_nets[net_type]
        with torch.no_grad():
            pre = net(x).argmax(dim=1)
            f_map = net.features(x)
            f_map_reshape = f_map.reshape(f_map.shape[0], f_map.shape[1], -1)
            weights = net.classifier[-1].weight.data
            weights = torch.softmax(weights,dim=1)
            weights = weights[pre].unsqueeze(1)
            heat = torch.bmm(weights,f_map_reshape)
            heat = heat.reshape(f_map.shape[0], 1, f_map.shape[-1], f_map.shape[-1])
            heat = torch.nn.functional.interpolate(heat, size=(32, 32), model="bilinear", align_corners=False)
            min_val = heat.min(dim=2, keepdim=True)[0].min(dim=3,keepdim=True)[0]
            max_val = heat.max(dim=2, keepdim=True)[0].max(dim=3,keepdim=True)[0]
            heats += (heat - min_val) / (max_val - min_val)
    return heats / len(classifier_nets)



def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    denormalized = tensor * std_tensor + mean_tensor
    return denormalized

def visualization(x,x_hat,e,c,epoch):
    save_path = "temp"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with torch.no_grad():
        x_grid = Image.fromarray((make_grid(denormalize(x[:16])).numpy()*255).astype(np.uint8)).save(f"{save_path}/x_{epoch}.png")
        x_hat_grid = Image.fromarray((make_grid(denormalize(x_hat[:16])).numpy()*255).astype(np.uint8)).save(f"{save_path}/x_hat_{epoch}.png")
        e = torch.abs(e)
        e_grid = Image.fromarray((make_grid(denormalize(e[:16])).numpy()*255).astype(np.uint8)).save(f"{save_path}/e_{epoch}.png")
        c_grid = Image.fromarray((make_grid(denormalize(c[:16])).numpy()*255).astype(np.uint8)).save(f"{save_path}/c_{epoch}.png")


def cal_psnr(net,classifier_nets,dataloader,device):
    print("计算PSNR...")
    psnr = []
    with torch.no_grad():
        for (x, _) in dataloader:
            x = x.to(device)
            c = get_cam(x,classifier_nets,device)
            e = net(x,c)
            x_hat = torch.clamp(x+e, min=-1.0, max=1.0)
            x = (denormalize(x).cpu().numpy() * 255).astype(np.uint8)
            x_hat = (denormalize(x_hat).cpu().numpy * 255).astype(np.uint8)
            psnr.append(peak_signal_noise_ratio(x,x_hat))
    return sum(psnr) / len(psnr)

def cal_rca(net,classifier_nets,dataloader,device):
    print("计算RCA...")
    n_sample = len(dataloader.dataset)
    rca = []
    with torch.no_grad():
        for net_type in classifier_nets.keys():
            c = 0
            classifier_net = classifier_nets[net_type]
            for (x, _) in dataloader:
                x = x.to(device)
                c = get_cam(x,classifier_nets,device)
                e = net(x,c)
                x_hat = torch.clamp(x+e, min=-1.0, max=1.0)
                Ln = classifier_net(x).argmax(dim=1)
                Ln_hat = classifier_net(x_hat).argmax(dim=1)
                c += (Ln==Ln_hat).sum().item()
            rca.append(c / n_sample)
    return sum(rca) / len(rca)
           

def train(net,classifier_nets,dataloader,loss_fn,optimizer,device,total_epochs):
    n = len(dataloader["train"])
    for epoch in range(1,total_epochs+1):
        net.train()
        l,l1,l2,l3 = 0.0,0.0,0.0,0.0
        for (x, _) in tqdm(dataloader["train"]):
            x = x.to(device)
            c = get_cam(x,classifier_nets,device)
            e = net(x,c)
            x_hat = torch.clamp(x+e, min=-1.0, max=1.0)
            optimizer.zero_grad()
            loss,loss1,loss2,loss3 = loss_fn(x,x_hat,c,e)
            l += loss
            l1 += loss1
            l2 += loss2
            l3 += loss3
            loss.backward()
            optimizer.step()
        psnr = cal_psnr(net,classifier_nets,dataloader["train"],device)
        rca = cal_rca(net,classifier_nets,dataloader["train"],device)
        print(f"[Epoch{epoch}/{total_epochs}][loss:{l/n} loss1:{l1/n} loss2:{l2/n} loss3:{l3/n} psnr:{psnr} RCA:{rca}]")
        if epoch in [1,15,45,145]:
            visualization(x,x_hat,e,c,epoch)
        if epoch % 10 == 0:
            torch.save(net,"dmv-jnd-{epoch}.pth")



def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    classifier_nets = {}
    net_path = "ckpts"
    # net_types = ["alexnet-GAP","vgg16-GAP","resnet50-GAP","densenet169-GAP"]
    net_types = ["alexnet-GAP"]
    nets = os.listdir(net_path)
    for net_type in net_types:
        model_name = [n for n in nets if n.startswith(net_type)][0]
        net = torch.load(f=f"{net_path}/{model_name}",map_location="cpu")
        net.eval()
        net.to(device)
        classifier_nets[net_type] = net
    net = JNDNet()
    loss_fn = Loss()
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-4,weight_decay=1e-3)
    dataloader = get_dataloader(batch_size=64,task="dmv-jnd")
    total_epochs = 3
    train(net,classifier_nets,dataloader,loss_fn,optimizer,device,total_epochs)
        
    

