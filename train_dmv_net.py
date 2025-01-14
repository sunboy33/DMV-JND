import torch.nn.functional
from torchvision import transforms
import torch



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








def train(net,classifier_nets,dataloader,loss_fn,optimizer,device,total_epochs):

    for epoch in range(1,total_epochs+1):
        net.train()
        for (x, _) in dataloader["train"]:
            x = x.to(device)
            c = get_cam(x,classifier_nets,device)
            e = net(x,c)
            x_hat = torch.clamp(x+e, min=-1.0, max=1.0)
            optimizer.zero_grad()
            loss,loss1,loss2,loss3 = loss_fn(x,x_hat,c,e)
            loss.backward()
            optimizer.step()


        

