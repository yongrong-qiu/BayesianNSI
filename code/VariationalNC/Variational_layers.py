import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

from VariationalNC.Variational_utils import Gaussian, ScaleMixtureGaussian
from VariationalNC.Variational_utils import VariationalModule


class VariationalLinear(VariationalModule):
    """
    Variational Linear layer
    """
    def __init__(self,\
                 in_features,\
                 out_features,\
                 PI = 0.2,\
                 SIGMA_1 =1,\
                 SIGMA_2 =0.00247875):  #0.04978706):  #torch.cuda.FloatTensor([np.exp(-6)])):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.PI = PI
        self.SIGMA_1 = SIGMA_1
        self.SIGMA_2 = SIGMA_2
        # Weight parameters
        #self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        #stdv = 1. / np.sqrt((self.in_features+self.out_features)/6)
        stdv = 1. / np.sqrt(self.in_features)
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-stdv, stdv))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5,-4))
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        #self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-0.2, 0.2))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).uniform_(-stdv, stdv))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0
    #
    def forward(self, input, sampleFlag=False, calculate_log_probs=False):
        if self.training or sampleFlag:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        #
        return F.linear(input, weight, bias)

class VariationalConv2d(VariationalModule):
    """
    Variational Conv2d layer
    """
    def __init__(self,\
                 in_channels,\
                 out_channels,\
                 kernel_size,\
                 stride=1,\
                 padding=0,\
                 dilation=1,\
                 groups=1,\
                 bias=True,\
                 PI = 0.5,\
                 SIGMA_1 = 1,\
                 SIGMA_2 = 0.00247875):#0.04978706): #0.00012341):  #torch.cuda.FloatTensor([np.exp(-6)])):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.PI = PI
        self.SIGMA_1 = SIGMA_1
        self.SIGMA_2 = SIGMA_2
        # Weight parameters
        #https://github.com/keras-team/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
        #stdv = 1. / np.sqrt((self.in_channels*self.kernel_size[0]*self.kernel_size[1]+self.out_channels)/6)
        stdv = 1. / np.sqrt(self.in_channels*self.kernel_size[0]*self.kernel_size[1])
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size).uniform_(-stdv, stdv))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels, in_channels, *self.kernel_size).uniform_(-5,-4)) #(-5,-4)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-stdv, stdv))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0
    def forward(self, input, sampleFlag=False, calculate_log_probs=False):
        if self.training or sampleFlag:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.conv2d(input, weight=weight, bias=bias, stride=self.stride, \
                        padding=self.padding, dilation=self.dilation, groups=self.groups)
    
    
class VariationalConv3d(VariationalModule):
    """
    Variational Conv3d layer
    """
    def __init__(self,\
                 in_channels,\
                 out_channels,\
                 kernel_size,\
                 stride=1,\
                 padding=0,\
                 dilation=1,\
                 groups=1,\
                 bias=True,\
                 PI = 0.5,\
                 SIGMA_1 = 1,\
                 SIGMA_2 = 0.00247875):#0.04978706): #0.00012341):  #torch.cuda.FloatTensor([np.exp(-6)])):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.PI = PI
        self.SIGMA_1 = SIGMA_1
        self.SIGMA_2 = SIGMA_2
        # Weight parameters
        #https://github.com/keras-team/keras/blob/998efc04eefa0c14057c1fa87cab71df5b24bf7e/keras/initializations.py
        #stdv = 1. / np.sqrt((self.in_channels*self.kernel_size[0]*self.kernel_size[1]+self.out_channels)/6)
        stdv = 1. / np.sqrt(self.in_channels*self.kernel_size[0]*self.kernel_size[1]*self.kernel_size[2])
        self.weight_mu = nn.Parameter(torch.Tensor(out_channels, in_channels,*self.kernel_size).uniform_(-stdv, stdv))
        self.weight_rho = nn.Parameter(torch.Tensor(out_channels,in_channels, *self.kernel_size).uniform_(-5,-4)) #(-5,-4)
        self.weight = Gaussian(self.weight_mu, self.weight_rho)
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_channels).uniform_(-stdv, stdv))
        self.bias_rho = nn.Parameter(torch.Tensor(out_channels).uniform_(-5,-4))
        self.bias = Gaussian(self.bias_mu, self.bias_rho)
        # Prior distributions
        self.weight_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.bias_prior = ScaleMixtureGaussian(self.PI, self.SIGMA_1, self.SIGMA_2)
        self.log_prior = 0
        self.log_variational_posterior = 0
    def forward(self, input, sampleFlag=False, calculate_log_probs=False):
        if self.training or sampleFlag:
            weight = self.weight.sample()
            bias = self.bias.sample()
        else:
            weight = self.weight.mu
            bias = self.bias.mu
        if self.training or calculate_log_probs:
            self.log_prior = self.weight_prior.log_prob(weight) + self.bias_prior.log_prob(bias)
            self.log_variational_posterior = self.weight.log_prob(weight) + self.bias.log_prob(bias)
        else:
            self.log_prior, self.log_variational_posterior = 0, 0
        return F.conv3d(input, weight=weight, bias=bias, stride=self.stride,\
                        padding=self.padding, dilation=self.dilation, groups=self.groups)

