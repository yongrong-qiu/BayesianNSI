import torch
import torch.nn as nn
#from torch.nn import functional as F

from VariationalNC.Variational_layers import VariationalLinear, VariationalConv2d, VariationalConv3d
from VariationalNC.Variational_utils import variational_estimator

@variational_estimator
class VariationalLN(nn.Module):
    def __init__(self):
        super().__init__()
        self.numofinput=50*40 #number of input
        self.numofneuron=23 #number of neurons
        #
        self.fc1= VariationalLinear(self.numofinput,self.numofneuron)
    #
    def forward(self, x, sampleFlag=False):
        encoded1 = x.view(-1,self.numofinput)
        encoded1 = torch.exp(self.fc1(encoded1, sampleFlag))
        return encoded1
