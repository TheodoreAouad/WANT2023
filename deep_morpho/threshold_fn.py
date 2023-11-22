import torch
from math import pi


def arctan_threshold(x):
    return 1/pi * torch.arctan(x) + 1/2


def tanh_threshold(x):
    return 1/2 * torch.tanh(x) + 1/2


def relu_threshold(x):
    return torch.relu(x)


def sigmoid_threshold(x):
    return torch.sigmoid(x)


def erf_threshold(x):
    return 1/2 * torch.erf(x) + 1/2


def clamp_threshold(x, s1=0, s2=1):
    return x.clamp(s1, s2)


def arctan_threshold_inverse(y):
    return torch.tan((y - 1/2) * pi)


def identity_threshold_inverse(y):
    return y


def relu_threshold_inverse(y):
    return y


def tanh_threshold_inverse(y):
    return torch.atanh((y - 1/2) * 2)


def sigmoid_threshold_inverse(y):
    return torch.logit(y)


def softplus_threshold_inverse(y, beta=1, threshold=10):
    output = y + 0
    output[beta * y < threshold] = 1 / beta * torch.log(torch.exp(beta * y[beta * y < threshold]) - 1)
    return output


def tanh_threshold_symetric(x):
    return torch.tanh(x)


def tanh_threshold_symetric_inverse(y):
    return torch.atanh(y)


def sigmoid_threshold_symetric(x):
    return 2 * torch.sigmoid(x) - 1


def sigmoid_threshold_symetric_inverse(y):
    return torch.logit((y + 1) / 2)


def arctan_threshold_symetric(x):
    return 2 * arctan_threshold(x) - 1


def arctan_threshold_symetric_inverse(y):
    return arctan_threshold_inverse((y + 1) / 2)


def erf_threshold_symetric(x):
    return torch.erf(x)

