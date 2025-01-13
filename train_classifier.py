import os
import torch
from models import get_net
from datasets import get_dataloader




def train(net,dataloader,loss_fn,optimizer,net_type,device,total_epochs):
    if not os.path.exists("ckpts"):
        os.mkdir("ckpts")
    train_samples = len(dataloader["train"].dataset)
    val_samples = len(dataloader["val"].dataset)
    tds = len(dataloader["train"])
    vds = len(dataloader["val"])
    best_v_acc = 0
  
    for epoch in range(1,total_epochs+1):
        net.train()
        t_loss = 0.0
        c = 0
        for (image,label) in dataloader["train"]:
            image,label = image.to(device),label.to(device)
            output = net(image)
            loss = loss_fn(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pre = output.argmax(dim=1)
            c += (pre == label).sum().item()
            t_loss += loss.item()
        net.eval()
        v_loss = 0.0
        v_c = 0
        with torch.no_grad():
            for (image,label) in dataloader["val"]:
                image,label = image.to(device),label.to(device)
                output = net(image)
                loss = loss_fn(output,label)
                pre = output.argmax(dim=1)
                v_c += (pre == label).sum().item()
                v_loss += loss.item()
        print(f"[Epoch{epoch}/{total_epochs}][t_loss:{t_loss/tds} t_acc:{c/train_samples} v_loss:{v_loss/vds} v_acc:{v_c/val_samples}]")
        v_acc = v_c / val_samples
        save_path = os.path.join("ckpts",f"{net_type}_best_acc_{v_acc:.3f}.pth")
        remove_path = os.path.join("ckpts",f"{net_type}_best_acc_{best_v_acc:.3f}.pth")
        if v_acc > best_v_acc:
            if os.path.exists(remove_path):
                os.remove(remove_path)
            torch.save(net,save_path)




def main():
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    num_classes = 10
    dataloader = get_dataloader(batch_size=64)
    loss_fn = torch.nn.CrossEntropyLoss()
    net_types = ["alexnet-GAP","vgg16-GAP","resnet50-GAP","densenet169-GAP"]
    total_epochs = 3
    for net_type in net_types:
        net = get_net(net_type,num_classes)
        optimizer = torch.optim.SGD(net.parameters(),lr=0.001,weight_decay=1e-4,momentum=0.9)
        train(net,dataloader,loss_fn,optimizer,net_type,device,total_epochs)

if __name__ == "__main__":
    main()