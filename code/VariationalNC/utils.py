import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


def adjust_spines(ax, spines):
    """
    axis spine
    use it like: adjust_spines(ax, ['left', 'bottom'])
    """
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(("outward", 2))  # outward by 2 points
            # spine.set_smart_bounds(True)
        else:
            spine.set_color("none")  # don't draw spine
    # turn off ticks where there is no spine
    if "left" in spines:
        ax.yaxis.set_ticks_position("left")
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])
    if "bottom" in spines:
        ax.xaxis.set_ticks_position("bottom")
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])


class Flatten(nn.Module):
    def forward(self, x):
        N, C, H, W = x.size()  # read in N, C, H, W
        return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """

    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)


class Flatten3D(nn.Module):
    def forward(self, x):
        N, C, D, H, W = x.size()  # read in N, C, D, H, W
        return x.view(N, -1)  # "flatten" the C *D* H * W values into a single vector per image


class Unflatten3D(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*D*H*W) and reshapes it
    to produce an output of shape (N, C, D,H, W).
    """

    def __init__(self, N=-1, C=128, D=7, H=7, W=7):
        super(Unflatten3D, self).__init__()
        self.N = N
        self.C = C
        self.D = D
        self.H = H
        self.W = W

    def forward(self, x):
        return x.view(self.N, self.C, self.D, self.H, self.W)


loss_mse = nn.MSELoss()
# loss_ssim = pytorch_ssim.SSIM(window_size=3)

# loss with L2 and L1 regularizer
# something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempN = x.shape[0]
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    l2temp = 0.0
    for temp in alpha_x:
        l2temp = l2temp + temp.weight.norm(2)
    L2loss = alpha * l2temp
    L1loss = beta * F.l1_loss(beta_y, torch.zeros_like(beta_y), reduction="sum")
    return (MSE + L2loss + L1loss) / (
        tempN * 10 * 2 * 56 * 56
    )  # batch size tempN, to be comparable with other ae variants loss


# loss with L2 and L1 regularizer, version2
# something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1v2(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempC, tempD, tempH, tempW = x.size()
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    l2temp = 0.0
    for temp in alpha_x:
        l2temp = l2temp + temp.weight.norm(2)
    L2loss = alpha * l2temp
    B, C, D, H, W = beta_y.size()  # Batch*channel*depth*height*width
    temp1 = beta_y.view(B, C, -1)
    temp2 = torch.norm(temp1, p=2, dim=2)
    temp3 = torch.sum(torch.abs(temp2))
    L1loss = beta * temp3
    # return (MSE+L2loss+L1loss)/(tempB* 2*10*12*12)#batch size tempN, to be comparable with other ae variants loss
    return (MSE + L2loss + L1loss) / (
        tempB * tempC * tempD * tempH * tempW
    )  # to be comparable with other ae variants loss


# loss with L2 and L1 regularizer, for supervised encoded
# something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN = x.size()
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    l2temp = 0.0
    for temp in alpha_x:
        l2temp = l2temp + temp.weight.norm(2)
    L2loss = alpha * l2temp
    l1temp = 0.0
    for temp in beta_y:
        l1temp = l1temp + temp.weight.norm(1)
    L1loss = beta * l1temp
    return (MSE + L2loss + L1loss) / (tempB * tempN)


def loss_L2L1_SE_regularizaion_3conv(recon_x, x, alpha1, alpha2, alpha3, beta, alpha_x1, alpha_x2, alpha_x3, beta_y):
    tempB, tempN = x.size()
    MSE = F.mse_loss(recon_x, x, reduction="sum")
    l2temp1 = 0.0
    l2temp2 = 0.0
    l2temp3 = 0.0
    for temp in alpha_x1:
        l2temp1 = l2temp1 + temp.weight.norm(2)
    for temp in alpha_x2:
        l2temp2 = l2temp2 + temp.weight.norm(2)
    for temp in alpha_x3:
        l2temp3 = l2temp3 + temp.weight.norm(2)
    L2loss = alpha1 * l2temp1 + alpha2 * l2temp2 + alpha3 * l2temp3
    l1temp = 0.0
    for temp in beta_y:
        l1temp = l1temp + temp.weight.norm(1)
    L1loss = beta * l1temp
    return (MSE + L2loss + L1loss) / (tempB * tempN)


# loss with L2 and L1 regularizer, for supervised encoded, Poisson loss
# something like loss = Poissonloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def Ploss_L2L1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN = x.size()
    Ploss = F.poisson_nll_loss(recon_x, x, log_input=False, reduction="sum")
    l2temp = 0.0
    for temp in alpha_x:
        l2temp = l2temp + temp.weight.norm(2)
    L2loss = alpha * l2temp
    l1temp = 0.0
    for temp in beta_y:
        l1temp = l1temp + temp.weight.norm(1)
    L1loss = beta * l1temp
    return Ploss + L2loss + L1loss  # (Ploss+L2loss+L1loss)/(tempB* tempN)


def Ploss_L2L1_SE_regularizaion_2conv(recon_x, x, alpha1, alpha2, beta, alpha_x1, alpha_x2, beta_y):
    tempB, tempN = x.size()
    Ploss = F.poisson_nll_loss(recon_x, x, log_input=False, reduction="sum")
    l2temp1 = 0.0
    l2temp2 = 0.0
    for temp in alpha_x1:
        l2temp1 = l2temp1 + temp.weight.norm(2)
    for temp in alpha_x2:
        l2temp2 = l2temp2 + temp.weight.norm(2)
    L2loss = alpha1 * l2temp1 + alpha2 * l2temp2
    l1temp = 0.0
    for temp in beta_y:
        l1temp = l1temp + temp.weight.norm(1)
    L1loss = beta * l1temp
    return Ploss + L2loss + L1loss  # (Ploss+L2loss+L1loss)/(tempB* tempN)


def Ploss_MAP(recon_x, x, log_prior, vbeta):
    tempB, tempN = x.size()
    Ploss = F.poisson_nll_loss(recon_x, x, log_input=False, reduction="sum")
    return -log_prior * vbeta + Ploss


"""
#loss with L2 and L1 regularizer, for supervised encoded, L2 for conv kernel smoothness
#something like loss = mseloss+ alpha*alpha_x(L2)+beta*beta_y(L1), here alpha_x is a list of model layers
def loss_L2lapL1_SE(recon_x, x, alpha, beta, alpha_x, beta_y):
    tempB, tempN =x.size()
    MSE = F.mse_loss(recon_x, x,reduction='sum')
    l2temp=0.0
    laplacian=torch.tensor([[0.5,1.0,0.5],[1.0,-6.0,1.0],[0.5,1.0,0.5]], requires_grad=False)#laplacian kernel
    for temp in alpha_x:
        #l2temp = l2temp+ temp.weight.norm(2)
        NN,CC=temp.weight.shape[0],temp.weight.shape[1]
        laplacians=laplacian.repeat(CC, CC, 1, 1).requires_grad_(False).to(device)
        temp2=F.conv2d(temp.weight,laplacians)
        l2temp = l2temp+ temp2.norm(2)
    L2loss=alpha*l2temp
    l1temp=0.0
    for temp in beta_y:
        l1temp = l1temp+ temp.weight.norm(1)
    L1loss=beta*l1temp
    return (MSE+L2loss+L1loss)/(tempB* tempN)
"""

# visualize DNN
# https://github.com/chongyangma/cs231n/blob/master/assignments/assignment3/NetworkVisualization-PyTorch.ipynb
# https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html
# https://jacobgil.github.io/deeplearning/filter-visualizations
# https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
# simple version: compute the gradient of the output channel wrt a blank image
# complex version: performa gradient ascend on the target channel, start with noise image
"""def vis_model_fl(model,device,xxshape):#visualize for final layer
    model=model.to(device)
    for param in model.parameters():
        param.requires_grad=False
    model=model.eval()
    (tempB,tempC,tempH,tempW)=xxshape#tempB should be equal to 1
    #xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
    xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
    if xx.grad is not None:
        xx.grad.data.zero_()
    out=model(xx)
    outlen=out.shape[1]
    yy=torch.zeros(outlen,tempC,tempH,tempW)
    for ii in range(outlen):
        if xx.grad is not None:
            xx.grad.data.zero_()
        out=model(xx)
        temp=out[0,ii]
        temp.backward()
        yy[ii]=xx.grad.data
        #if xx.grad is not None:
        #    xx.grad.data.zero_()
    return yy"""


def vis_model_fl(model, device, xxshape, test_samples=100):  # test_samples: for bayesian model
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False
    model = model.eval()
    if "3d" in model.__class__.__name__:
        (tempB, tempC, tempD, tempH, tempW) = xxshape  # tempB should be equal to 1
        # xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
        # xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
        xx = torch.zeros((tempB, tempC, tempD, tempH, tempW)).to(device)
        xx.requires_grad = True
        if xx.grad is not None:
            xx.grad.data.zero_()
        if "Variational" in model.__class__.__name__:
            out = model(xx, sampleFlag=False)
            outlen = out.shape[1]
            yy = torch.zeros(test_samples + 1, outlen, tempC, tempD, tempH, tempW)
            for one_sample in range(test_samples):
                for ii in range(outlen):
                    if xx.grad is not None:
                        xx.grad.data.zero_()
                    out = model(xx, sampleFlag=True)
                    temp = out[0, ii]
                    temp.backward()
                    yy[one_sample, ii] = xx.grad.data
            for ii in range(outlen):
                if xx.grad is not None:
                    xx.grad.data.zero_()
                out = model(xx, sampleFlag=False)
                temp = out[0, ii]
                temp.backward()
                yy[test_samples, ii] = xx.grad.data
        else:  # vanilla model
            out = model(xx)
            outlen = out.shape[1]
            yy = torch.zeros(outlen, tempC, tempD, tempH, tempW)
            for ii in range(outlen):
                if xx.grad is not None:
                    xx.grad.data.zero_()
                out = model(xx)
                temp = out[0, ii]
                temp.backward()
                yy[ii] = xx.grad.data
                # if xx.grad is not None:
                #    xx.grad.data.zero_()

    else:
        (tempB, tempC, tempH, tempW) = xxshape  # tempB should be equal to 1
        # xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
        # xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
        xx = torch.zeros((tempB, tempC, tempH, tempW)).to(device)
        xx.requires_grad = True
        if xx.grad is not None:
            xx.grad.data.zero_()
        if "Variational" in model.__class__.__name__:
            out = model(xx, sampleFlag=False)
            outlen = out.shape[1]
            yy = torch.zeros(test_samples + 1, outlen, tempC, tempH, tempW)
            for one_sample in range(test_samples):
                for ii in range(outlen):
                    if xx.grad is not None:
                        xx.grad.data.zero_()
                    out = model(xx, sampleFlag=True)
                    temp = out[0, ii]
                    temp.backward()
                    yy[one_sample, ii] = xx.grad.data
            for ii in range(outlen):
                if xx.grad is not None:
                    xx.grad.data.zero_()
                out = model(xx, sampleFlag=False)
                temp = out[0, ii]
                temp.backward()
                yy[test_samples, ii] = xx.grad.data
        else:  # vanilla model
            out = model(xx)
            outlen = out.shape[1]
            yy = torch.zeros(outlen, tempC, tempH, tempW)
            for ii in range(outlen):
                if xx.grad is not None:
                    xx.grad.data.zero_()
                out = model(xx)
                temp = out[0, ii]
                temp.backward()
                yy[ii] = xx.grad.data
                # if xx.grad is not None:
                #    xx.grad.data.zero_()
    return yy


def enable_dropout1(model):
    """Function to enable the dropout layers during test-time"""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def vis_model_fl_dropout(model, device, xxshape, dropout_num=100):  # test_samples: for bayesian model
    model = model.to(device)
    model = model.eval()
    enable_dropout1(model)
    kk = 0
    # enable_dropout1(model)
    for param in model.parameters():
        param.requires_grad = False

    if "3d" in model.__class__.__name__:
        (tempB, tempC, tempD, tempH, tempW) = xxshape  # tempB should be equal to 1
        # xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
        # xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
        xx = torch.zeros((tempB, tempC, tempD, tempH, tempW)).to(device)
        xx.requires_grad = True
        if xx.grad is not None:
            xx.grad.data.zero_()
        if "Dropout" in model.__class__.__name__:
            out = model(xx)
            outlen = out.shape[1]
            yy = torch.zeros(dropout_num, outlen, tempC, tempD, tempH, tempW)
            for one_sample in range(dropout_num):
                for ii in range(outlen):
                    if xx.grad is not None:
                        xx.grad.data.zero_()
                    out = model(xx)
                    temp = out[0, ii]
                    temp.backward()
                    yy[one_sample, ii] = xx.grad.data
        else:  # vanilla model
            out = model(xx)
            outlen = out.shape[1]
            yy = torch.zeros(outlen, tempC, tempD, tempH, tempW)
            for ii in range(outlen):
                if xx.grad is not None:
                    xx.grad.data.zero_()
                out = model(xx)
                temp = out[0, ii]
                temp.backward()
                yy[ii] = xx.grad.data

    else:
        (tempB, tempC, tempH, tempW) = xxshape  # tempB should be equal to 1
        # xx=torch.randn((tempB,tempC,tempH,tempW),requires_grad=True)
        # xx=torch.zeros((tempB,tempC,tempH,tempW),requires_grad=True)
        xx = torch.zeros((tempB, tempC, tempH, tempW)).to(device)
        xx.requires_grad = True
        if xx.grad is not None:
            xx.grad.data.zero_()
        if "Dropout" in model.__class__.__name__:
            out = model(xx)
            outlen = out.shape[1]
            yy = torch.zeros(dropout_num, outlen, tempC, tempH, tempW)
            for one_sample in range(dropout_num):
                for ii in range(outlen):
                    if xx.grad is not None:
                        xx.grad.data.zero_()
                    out = model(xx)
                    temp = out[0, ii]
                    temp.backward()
                    yy[one_sample, ii] = xx.grad.data
        else:  # vanilla model
            out = model(xx)
            outlen = out.shape[1]
            yy = torch.zeros(outlen, tempC, tempH, tempW)
            for ii in range(outlen):
                if xx.grad is not None:
                    xx.grad.data.zero_()
                out = model(xx)
                temp = out[0, ii]
                temp.backward()
                yy[ii] = xx.grad.data
    return yy

