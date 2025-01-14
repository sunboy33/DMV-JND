
import torch.nn as nn



class Loss(nn.Module):
    def __init__(self):
        super().__init__()
    

    def loss1(self,x,x_hat,classifier_nets):
        pass

    def loss2(self,c,e):
        pass


    def loss3(self,c,e):
        pass


    def forward(self,x,x_hat,c,e,classifier_nets):
        alpha,beta = 1.0,1.0
        loss1 = self.loss1(x,x_hat,classifier_nets)
        loss2 = self.loss2(c,e)
        loss3 = self.loss2(c,e)
        loss = loss1 + alpha * loss2 + beta * loss3
        return loss