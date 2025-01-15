import torch
import torch.nn as nn
import torch.nn.functional as F



class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
    

    def loss1(self,x,x_hat,classifier_nets):
        x = F.interpolate(x,size=(224,224),mode="bilinear",align_corners=False)
        x_hat = F.interpolate(x_hat,size=(224,224),mode="bilinear",align_corners=False)
        loss = 0.0
        for net_type in classifier_nets.keys():
            net = classifier_nets[net_type]
            Sn = net(x)
            Sn_hat = net(x_hat)
            Ln = Sn.argmax(dim=1)
            loss += self.ce(Sn_hat,Ln)
        return loss / len(classifier_nets)

    def loss2(self,c,e):
        I = c.mean(dim=(1,2,3))
        N = 1 - I
        e = torch.abs(e)
        e = e[:,0] / 3 + e[:,1] / 3 + e[:,2] / 3
        N0 = e.mean(dim=(1,2))
        q = 1e-10
        loss = torch.log((N**2 + N0**2 + q) / (2 * N * N0 + q))
        return loss.mean()


    def loss3(self,c,e):
        bs = c.shape[0]
        c = c.reshape(bs, -1)
        v = torch.softmax(c, dim=1)
        e = torch.abs(e)
        e = e[:,0] / 3 + e[:,1] / 3 + e[:,2] / 3
        e = e.reshape(bs, -1)
        loss = torch.sum(v * e, dim=1)
        return loss.mean()


    def forward(self,x,x_hat,c,e,classifier_nets):
        alpha,beta = 1.0,1.0
        loss1 = self.loss1(x,x_hat,classifier_nets)
        loss2 = self.loss2(c,e)
        loss3 = self.loss2(c,e)
        loss = loss1 + alpha * loss2 + beta * loss3
        return loss