import os
import torch
from torch import nn
from torch import distributions

# FrEIA (https://github.com/VLL-HD/FrEIA/)
import FrEIA.framework as Ff
import FrEIA.modules as Fm

from flow_model.subnet import *
from config import WEIGHT_DIR


def get_flow_model(c):
    dims_in = [c.channels, int(c.img_size[0] / c.scales), int(c.img_size[1] / c.scales)]
    nodes = Ff.SequenceINN(*dims_in)
    for i in range(c.coupling * 2):
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=get_subnet(c=c),
            affine_clamping=c.clamp,
            permute_soft=False,
        )
    return nodes


class RessubFlow(nn.Module):

    def __init__(self, c):
        super(RessubFlow, self).__init__()
        self.flow_model = get_flow_model(c)

        # isotropic standard normal distribution
        self.prior = distributions.Normal(torch.tensor(0.).to(c.device), torch.tensor(1.).to(c.device))
        self.negative_val = c.negative_val
        self.neg_loss_weight = c.neg_loss_weight

    def forward(self, images):
        z, log_jac_dets = self.flow_model(images)

        return z, log_jac_dets

    def get_loss(self, B, z, log_jac_dets):
        loss_positive, loss_negative = self.get_neg_log_likelihood(B, z, log_jac_dets)
        max_loss_positive = loss_positive.max()

        loss_negative = loss_negative[loss_negative < self.negative_val]
        loss_negative = loss_negative[loss_negative < max_loss_positive]

        if loss_negative.shape[0] > 0:
            loss_negative *= (-1)
            loss_negative = loss_negative.mean()
            loss_positive = loss_positive.mean()
            loss = (1 - self.neg_loss_weight) * loss_positive + self.neg_loss_weight * loss_negative
        else:
            loss_negative = torch.tensor(0.)
            loss_positive = loss_positive.mean()
            loss = loss_positive
        return loss, loss_positive, loss_negative

    def get_neg_log_likelihood(self, B, z, log_jac_dets):
        log_prior_prob = torch.sum(self.prior.log_prob(z), dim=(1, 2, 3))
        log_likelihood = log_prior_prob + log_jac_dets
        neg_log_likelihood = -log_likelihood
        return neg_log_likelihood[:B], neg_log_likelihood[B:]


def save_model(model, filename, type):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)

    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, f'{filename}_{type}'))


def load_model(model, filename):
    state = torch.load(filename)
    model.load_state_dict(state, strict=True)
    return model
