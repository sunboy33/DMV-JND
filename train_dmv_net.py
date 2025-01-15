import os
import random
import argparse
import numpy as np
import torch
import torch.nn.functional
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio
from PIL import Image
from models import JNDNet
from loss import Loss
from datasets import get_dataloader


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--num_workers',type=int,default=2)
    parser.add_argument('--weight_decay',type=float,default=1e-3)
    parser.add_argument('--total_epochs',type=int,default=200)
    parser.add_argument('--net_types',type=str,nargs='*',default="alexnet-GAP")
    parser.add_argument('--n',type=int,default=16)
    return parser.parse_args()

args = get_parser()

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
            heat = torch.nn.functional.interpolate(heat, size=(32, 32), mode="bilinear", align_corners=False)
            min_val = heat.min(dim=2, keepdim=True)[0].min(dim=3,keepdim=True)[0]
            max_val = heat.max(dim=2, keepdim=True)[0].max(dim=3,keepdim=True)[0]
            heats += (heat - min_val) / (max_val - min_val)
    return heats / len(classifier_nets)



def denormalize(tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)
    denormalized = tensor * std_tensor + mean_tensor
    return denormalized

def visualization(x,x_hat,e,c,epoch,n):
    def save(tensor,epoch,type):
        if type == "c":
            tensor_grid = make_grid(tensor[:n].cpu()).numpy()
            Image.fromarray(np.array(tensor_grid[0] * 255,dtype=np.uint8)).save(f"{save_path}/{epoch}_{type}.png")
        else:
            tensor_grid = make_grid(denormalize(tensor[:n].cpu())).numpy()
            Image.fromarray(np.array(tensor_grid.transpose(1, 2, 0) * 255,dtype=np.uint8)).save(f"{save_path}/{epoch}_{type}.png")
    save_path = "temp"
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with torch.no_grad():
        save(x,epoch,"x")
        save(x_hat,epoch,"x_hat")
        e = torch.abs(e)
        save(e,epoch,"e")
        save(c,epoch,"c")
        


def cal_psnr(net,classifier_nets,dataloader,device):
    psnr = []
    with torch.no_grad():
        for (x, _) in tqdm(dataloader,desc="PSNR"):
            x = x.to(device)
            c = get_cam(x,classifier_nets,device)
            e = net(x,c)
            x_hat = torch.clamp(x+e, min=-1.0, max=1.0)
            x = (denormalize(x.cpu()).numpy() * 255).astype(np.uint8)
            x_hat = (denormalize(x_hat.cpu()).numpy() * 255).astype(np.uint8)
            psnr.append(peak_signal_noise_ratio(x,x_hat))
    return sum(psnr) / len(psnr)

def cal_rca(net,classifier_nets,dataloader,device):
    n_sample = len(dataloader.dataset)
    rca = []
    with torch.no_grad():
        for net_type in classifier_nets.keys():
            count = 0
            classifier_net = classifier_nets[net_type]
            for (x, _) in tqdm(dataloader,desc=f"{net_type} RCA"):
                x = x.to(device)
                c = get_cam(x,classifier_nets,device)
                e = net(x,c)
                x_hat = torch.clamp(x+e, min=-1.0, max=1.0)
                Ln = classifier_net(x).argmax(dim=1)
                Ln_hat = classifier_net(x_hat).argmax(dim=1)
                count += (Ln==Ln_hat).sum().item()
            rca.append(count / n_sample)
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
            loss,loss1,loss2,loss3 = loss_fn(x,x_hat,c,e,classifier_nets)
            l += loss.item()
            l1 += loss1.item()
            l2 += loss2.item()
            l3 += loss3.item()
            loss.backward()
            optimizer.step()
        psnr = cal_psnr(net,classifier_nets,dataloader["train"],device)
        rca = cal_rca(net,classifier_nets,dataloader["train"],device)
        print(f"[Epoch{epoch}/{total_epochs}][loss:{l/n:.3f} loss1:{l1/n:.3f} loss2:{l2/n:.3f} loss3:{l3/n:.3f} psnr:{psnr:.3f} RCA:{rca:.3f}]")
        if epoch in [1,15,45,145]:
            visualization(x,x_hat,e,c,epoch,args.n)
        if epoch % 10 == 0:
            torch.save(net,"dmv-jnd-{epoch}.pth")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_init_fn(worker_id):
    np.random.seed(args.seed + worker_id)
    random.seed(args.seed + worker_id)

def main():
    
    set_seed(args.seed)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    classifier_nets = {}
    net_path = "ckpts"
    # net_types = ["alexnet-GAP","vgg16-GAP","resnet50-GAP","densenet169-GAP"]
    nets = os.listdir(net_path)
    for net_type in args.net_types:
        model_name = [n for n in nets if n.startswith(net_type)][0]
        net = torch.load(f=f"{net_path}/{model_name}",map_location="cpu")
        net.eval()
        net.to(device)
        classifier_nets[net_type] = net
    net = JNDNet()
    net.to(device)
    loss_fn = Loss()
    optimizer = torch.optim.Adam(net.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    dataloader = get_dataloader(batch_size=args.batch_size,num_workers=args.num_workers,task="dmv-jnd",worker_init_fn=worker_init_fn)
    train(net,classifier_nets,dataloader,loss_fn,optimizer,device,args.total_epochs)
    

if __name__== "__main__":
    main()
    

