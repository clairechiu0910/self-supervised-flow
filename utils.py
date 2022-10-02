import os
import torch

from config import RESULT_DIR, VERBOSE_DIR


class Score_Observer:
    '''Keeps an eye on the current and highest score so far'''

    def __init__(self, c, name):
        self.name = name
        self.class_name = c.class_name
        self.model_name = c.model_name

        self.max_epoch = 0
        self.max_score = None
        self.last = None
        self.verbose = c.verbose

    def update(self, score, epoch, print_score=False):
        self.last = score
        if epoch == 0 or score > self.max_score:
            self.max_score = score
            self.max_epoch = epoch
        if print_score:
            self.print_score()

    def print_score(self):
        result_msg = 'last: {:6.4f} \t max: {:6.4f} \t epoch_max: {:3d} \t {:s}::{:s}'.format(
            self.last, self.max_score, self.max_epoch, self.class_name, self.name)

        if self.verbose:
            write_verbose(self.model_name, result_msg)

        fp = open(os.path.join(os.path.join(RESULT_DIR, '{}.txt'.format(self.model_name))), 'a')
        fp.write(result_msg + '\n')
        fp.close()


class UnNormalize(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


def t2np(tensor):
    '''pytorch tensor -> numpy array'''
    return tensor.cpu().data.numpy() if tensor is not None else None


def flat(tensor):
    return tensor.reshape(tensor.shape[0], -1)


def cat_maps(z):
    return torch.cat([z[i].reshape(z[i].shape[0], -1) for i in range(len(z))], dim=1)


def write_verbose(model_name, msg):
    fp = open(os.path.join(os.path.join(VERBOSE_DIR, '{}.txt'.format(model_name))), 'a')
    fp.write(msg + '\n')
    fp.close()


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())
