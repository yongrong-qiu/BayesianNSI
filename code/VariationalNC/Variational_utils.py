import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


#function for Variational

class VariationalModule(nn.Module):
    """
    creates base class for VariationalNN, in order to enable specific behavior
    """
    def init(self):
        super().__init__()

'''
class Gaussian(object):
    def __init__(self, mu, rho):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.normal = torch.distributions.Normal(0,1)
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    def sample(self):
        epsilon = self.normal.sample(self.rho.size()).to(device)
        return self.mu + self.sigma * epsilon
    def log_prob(self, input):
        return (-np.log(np.sqrt(2 * np.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
'''
class Gaussian(nn.Module):
    """
    posterior probability of weights, trainable, for MC sampling weights
    """
    def __init__(self, mu, rho):
        super().__init__()
        #self.mu = mu
        #self.rho = rho
        self.mu = nn.Parameter(mu)
        self.rho = nn.Parameter(rho)
        #self.normal = torch.distributions.Normal(0,1)
        self.register_buffer('epsilon', torch.zeros(self.rho.size()))
    @property
    def sigma(self):
        return torch.log1p(torch.exp(self.rho))
    def sample(self):
        self.epsilon.data.normal_()
        #epsilon = self.normal.sample(self.rho.size()) #.to(device)
        return self.mu + self.sigma * self.epsilon
    def log_prob(self, input):
        return (-np.log(np.sqrt(2 * np.pi))
                - torch.log(self.sigma)
                - ((input - self.mu) ** 2) / (2 * self.sigma ** 2)).sum()
#
'''
class ScaleMixtureGaussian(object):
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()
'''
class ScaleMixtureGaussian(nn.Module):
    """
    Scale mixture Gaussian as prior for weights, not learnable
    """
    def __init__(self, pi, sigma1, sigma2):
        super().__init__()
        self.pi = pi
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.gaussian1 = torch.distributions.Normal(0,sigma1)
        self.gaussian2 = torch.distributions.Normal(0,sigma2)
    def log_prob(self, input):
        prob1 = torch.exp(self.gaussian1.log_prob(input))
        prob2 = torch.exp(self.gaussian2.log_prob(input))
        return (torch.log(self.pi * prob1 + (1-self.pi) * prob2)).sum()

#decorator
def variational_estimator(nn_class):
    """
    This decorator adds some methods to a nn.Module, in order to facilitate the handling of
    Variational NN models
    
    Parameters:
        nn_class: torch.nn.Module 
        
    Returns a nn.Module with methods for:
        (1) Calculate prior loss for VariationalModules;
        (2) Calculate posterior loss for VariationalModules
        (3) Sample the Elbo Loss along its variational inferences (helps training)
        
    """
    def log_prior(self):
        log_priors=0
        for module in self.modules():
            if isinstance(module, (VariationalModule)):
                log_priors +=module.log_prior
        return log_priors
    setattr(nn_class, "log_prior", log_prior)
    #
    def log_variational_posterior(self):
        log_variational_posteriors=0
        for module in self.modules():
            if isinstance(module, (VariationalModule)):
                log_variational_posteriors +=module.log_variational_posterior
        return log_variational_posteriors
    setattr(nn_class, "log_variational_posterior", log_variational_posterior)
    #
    def sample_elbo(self, input, target, vbeta=1,sample_num=2):
        """
        Sample the Elbo
        
        Parameters:
            vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
            kl divergence and negative log likelihood
        """
        negative_log_likelihood=0
        log_priors=0
        log_variational_posteriors=0
        for _ in range(sample_num):
            outputs = self(input, sampleFlag=True)
            negative_log_likelihood += F.poisson_nll_loss(outputs, target,log_input=False, reduction='sum') 
            log_priors += self.log_prior()
            log_variational_posteriors +=self.log_variational_posterior()
        loss = ((log_variational_posteriors - log_priors)*vbeta + negative_log_likelihood)/sample_num
        negative_log_likelihood = negative_log_likelihood/sample_num
        log_priors = log_priors/sample_num
        log_variational_posteriors = log_variational_posteriors/sample_num
        return loss, log_priors, log_variational_posteriors, negative_log_likelihood
    setattr(nn_class, "sample_elbo", sample_elbo)
    #
    def map(self, input, target):
        """
        Maximize a posterior
        """
        outputs = self(input, sampleFlag=False)
        negative_log_likelihood = F.poisson_nll_loss(outputs, target,log_input=False, reduction='sum') 
        log_priors = self.log_prior()
        loss = - log_priors + negative_log_likelihood
        return loss, log_priors, negative_log_likelihood
    setattr(nn_class, "map", map)
    #    
    def sample_elbo_mse(self, input, target, vbeta=1,sample_num=1):
        """
        Sample the Elbo
        
        Parameters:
            vbeta: just like beta-VAE, this is a v(ariational)beta to control the ratio between
            kl divergence and negative log likelihood
        """
        negative_log_likelihood=0
        log_priors=0
        log_variational_posteriors=0
        for _ in range(sample_num):
            outputs = self(input, sampleFlag=True)
            #negative_log_likelihood += F.poisson_nll_loss(outputs, target,log_input=False, reduction='sum') 
            negative_log_likelihood += F.mse_loss(outputs, target,reduction='sum')
            log_priors += self.log_prior()
            log_variational_posteriors +=self.log_variational_posterior()
        #print('vbeta: {}'.format(vbeta))
        loss = ((log_variational_posteriors - log_priors)*vbeta + negative_log_likelihood)/sample_num
        negative_log_likelihood = negative_log_likelihood/sample_num
        log_priors = log_priors/sample_num
        log_variational_posteriors = log_variational_posteriors/sample_num
        #print('loss: {}'.format(loss))
        return loss, log_priors, log_variational_posteriors, negative_log_likelihood
    setattr(nn_class, "sample_elbo_mse", sample_elbo_mse)
    #
    return nn_class