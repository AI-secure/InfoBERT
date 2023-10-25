import numpy as np

import torch
import torch.nn as nn
from jutils import *

## cubic
# lowersize = 40
# hiddensize = 6

## Gaussian
# lowersize = 20
# hiddensize = 8

## club vs l1out
lowersize = 40
hiddensize = 8


class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, lr=1e-3, beta=0):
        super(CLUB, self).__init__()
        self.hiddensize = y_dim
        self.version = 0
        self.p_mu = nn.Sequential(nn.Linear(x_dim, self.hiddensize),
                                  nn.ReLU(),
                                  nn.Linear(self.hiddensize, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, self.hiddensize),
                                      nn.ReLU(),
                                      nn.Linear(self.hiddensize, y_dim),
                                      nn.Tanh())

        self.optimizer = torch.optim.Adam(self.parameters(), lr)
        self.beta = beta

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def mi_est_sample(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        sample_size = x_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()
        negative = - (mu - y_samples[random_index]) ** 2 / 2. / logvar.exp()
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = - (mu - y_samples) ** 2 / 2. / logvar.exp()

        prediction_1 = mu.unsqueeze(1)  # [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2. / logvar.exp()  # [nsample, dim]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean(), positive.sum(dim = -1).mean(), negative.sum(dim = -1).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)

        # return -1./2. * ((mu - y_samples)**2 /logvar.exp()-logvar ).sum(dim=1).mean(dim=0)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)

    def update(self, x_samples, y_samples):
        if self.version == 0:
            self.train()
            loss = - self.loglikeli(x_samples, y_samples)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            # self.eval()
            return self.mi_est_sample(x_samples, y_samples) * self.beta

        elif self.version == 1:
            self.train()
            x_samples = torch.reshape(x_samples, (-1, x_samples.shape[-1]))
            y_samples = torch.reshape(y_samples, (-1, y_samples.shape[-1]))

            loss = -self.loglikeli(x_samples, y_samples)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            upper_bound = self.mi_est_sample(x_samples, y_samples) * self.beta
            # self.eval()
            return upper_bound


class CLUBv2(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    def __init__(self, x_dim, y_dim, lr=1e-3, beta=0):
        super(CLUBv2, self).__init__()
        self.hiddensize = y_dim
        self.version = 2
        self.beta = beta

    def mi_est_sample(self, x_samples, y_samples):
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        positive = torch.zeros_like(y_samples)
        negative = - (y_samples - y_samples[random_index]) ** 2 / 2.
        upper_bound = (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return upper_bound/2.
        return upper_bound

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        positive = torch.zeros_like(y_samples)

        prediction_1 = y_samples.unsqueeze(1)  # [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        negative = - ((y_samples_1 - prediction_1) ** 2).mean(dim=1) / 2.   # [nsample, dim]
        return (positive.sum(dim=-1) - negative.sum(dim=-1)).mean()
        # return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean(), positive.sum(dim = -1).mean(), negative.sum(dim = -1).mean()

    def loglikeli(self, x_samples, y_samples):
        return 0

    def update(self, x_samples, y_samples, steps=None):
        # no performance improvement, not enabled
        if steps:
            beta = self.beta if steps > 1000 else self.beta * steps / 1000  # beta anealing
        else:
            beta = self.beta

        return self.mi_est_sample(x_samples, y_samples) * self.beta



class InfoNCE(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(InfoNCE, self).__init__()
        self.lower_size = 300
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, self.lower_size),
                                    nn.ReLU(),
                                    nn.Linear(self.lower_size, 1),
                                    nn.Softplus())

    def forward(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        random_index = torch.randint(sample_size, (sample_size,)).long()

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1))  # [s_size, s_size, 1]

        lower_bound = T0.mean() - (
                    T1.logsumexp(dim=1).mean() - np.log(sample_size))  # torch.log(T1.exp().mean(dim = 1)).mean()

        # compute the negative loss (maximise loss == minimise -loss)
        return lower_bound


class NWJ(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(NWJ, self).__init__()
        self.F_func = nn.Sequential(nn.Linear(x_dim + y_dim, lowersize),
                                    nn.ReLU(),
                                    nn.Linear(lowersize, 1))

    def mi_est(self, x_samples, y_samples):  # samples have shape [sample_size, dim]
        # shuffle and concatenate
        sample_size = y_samples.shape[0]
        # random_index = torch.randint(sample_size, (sample_size,)).long()

        x_tile = x_samples.unsqueeze(0).repeat((sample_size, 1, 1))
        y_tile = y_samples.unsqueeze(1).repeat((1, sample_size, 1))

        T0 = self.F_func(torch.cat([x_samples, y_samples], dim=-1))
        T1 = self.F_func(torch.cat([x_tile, y_tile], dim=-1)) - 1.  # [s_size, s_size, 1]

        lower_bound = T0.mean() - (T1.logsumexp(dim=1) - np.log(sample_size)).exp().mean()
        return lower_bound



class L1OutUB(nn.Module):  # naive upper bound
    def __init__(self, x_dim, y_dim):
        super(L1OutUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hiddensize),
                                  nn.ReLU(),
                                  nn.Linear(hiddensize, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hiddensize),
                                      nn.ReLU(),
                                      nn.Linear(hiddensize, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        batch_size = y_samples.shape[0]
        mu, logvar = self.get_mu_logvar(x_samples)

        positive = (- (mu - y_samples) ** 2 / 2. / logvar.exp() - logvar / 2.).sum(dim=-1)  # [nsample]

        mu_1 = mu.unsqueeze(1)  # [nsample,1,dim]
        logvar_1 = logvar.unsqueeze(1)
        y_samples_1 = y_samples.unsqueeze(0)  # [1,nsample,dim]
        all_probs = (- (y_samples_1 - mu_1) ** 2 / 2. / logvar_1.exp() - logvar_1 / 2.).sum(
            dim=-1)  # [nsample, nsample]

        # diag_mask = torch.ones([batch_size, batch_size,1]).cuda() - torch.ones([batch_size]).diag().unsqueeze(-1).cuda()
        diag_mask = torch.ones([batch_size]).diag().unsqueeze(-1).cuda() * (-20.)

        # negative = (all_probs + diag_mask).logsumexp(dim = 0) - np.log(y_samples.shape[0]-1.) #[nsample]
        inpt = all_probs + diag_mask
        negative = log_sum_exp(all_probs + diag_mask, dim=0) - np.log(y_samples.shape[0] - 1.)  # [nsample]
        return (positive - negative).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        # return -1./2. * ((mu - y_samples)**2 /logvar.exp()-logvar ).sum(dim=1).mean(dim=0)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)


class VarUB(nn.Module):  # variational upper bound
    def __init__(self, x_dim, y_dim):
        super(VarUB, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hiddensize),
                                  nn.ReLU(),
                                  nn.Linear(hiddensize, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hiddensize),
                                      nn.ReLU(),
                                      nn.Linear(hiddensize, y_dim),
                                      nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar

    def mi_est(self, x_samples, y_samples):  # [nsample, 1]
        mu, logvar = self.get_mu_logvar(x_samples)
        return 1. / 2. * (mu ** 2 + logvar.exp() - 1. - logvar).mean()

    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        # return -1./2. * ((mu - y_samples)**2 /logvar.exp()-logvar ).sum(dim=1).mean(dim=0)
        return (-(mu - y_samples) ** 2 / logvar.exp() - logvar).sum(dim=1).mean(dim=0)
