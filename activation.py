import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

def get_activation(
        activation_fn:str,
        param: float = 1.,
        dim: int = 1
):
    """
    Select the corresponding activation class
    Parameters
    ----------
    activation_fn
    t_invert
    beta
    power
    dim

    Returns
    -------

    """
    if activation_fn == 'triangle':
        return Triangle(power=param)
    if activation_fn == 'relu':
        return nn.ReLU()
    if activation_fn == 'repu':
        return RePU(power=param)
    if activation_fn == 'sigmoid':
        return Sigmoid(beta=param)
    if activation_fn == 'tanh':
        return Tanh(beta=param)
    if activation_fn == 'exp':
        return Exp(t_invert=param)
    if activation_fn == 'softmax':
        return SoftMax(t_invert=param, dim=dim)
    if activation_fn == 'hard':
        return Hard()

class RePU(nn.Module):
    r"""Applies the Repu function element-wise:
    """

    def __init__(self, power: float, inplace: bool = False):
        super(RePU, self).__init__()
        self.power = power
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return F.relu(input, inplace=self.inplace)**self.power

    def extra_repr(self) -> str:
        return 'power=%s'%self.power

class Tanh(nn.Module):
    r"""Applies the Tanh element-wise function:
    """
    def __init__(self, beta: float):
        super(Tanh, self).__init__()
        self.beta = beta

    def forward(self, input: Tensor) -> Tensor:
        return torch.tanh(input * self.beta)

    def extra_repr(self) -> str:
        return 'beta=%s'%self.beta

class Sigmoid(nn.Module):
    r"""Applies the Sigmoid element-wise function:
    """
    def __init__(self, beta: float):
        super(Sigmoid, self).__init__()
        self.beta = 10#beta

    def forward(self, input: Tensor) -> Tensor:
        return torch.sigmoid(input * self.beta)

    def extra_repr(self) -> str:
        return 'beta=%s'%self.beta

class Triangle(nn.Module):
    r"""Applies the Sigmoid element-wise function:
    """

    def __init__(self,  power: float=1, inplace: bool = True):
        super(Triangle, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input: Tensor) -> Tensor:
        input = input - torch.mean(input.data, axis=1, keepdims=True)
        return F.relu(input, inplace=self.inplace) ** self.power

    def extra_repr(self) -> str:
        return 'power=%s'%self.power


class Exp(nn.Module):
    r"""Applies the exp element-wise function:
    """
    def __init__(self, t_invert: float):
        super(Exp, self).__init__()
        self.t_invert = t_invert
    def forward(self, input: Tensor) -> Tensor:
        return torch.exp(input * self.t_invert)

    def extra_repr(self) -> str:
        return 't_invert=%s'%self.t_invert

class Hard(nn.Module):
    r"""Applies the exp element-wise function:
    """
    def __init__(self,):
        super(Hard, self).__init__()
    def forward(self, input: Tensor) -> Tensor:
        return  nn.functional.one_hot(input.argmax(dim=1), num_classes=input.shape[1]).to(
            torch.float)


class SoftMax(nn.Module):
    r"""Applies the softmax function element-wise:
    """

    def __init__(self, t_invert: float, dim: Union[int, tuple] = 1):
        super(SoftMax, self).__init__()
        self.t_invert = t_invert
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(self.dim, int):
            return torch.softmax(self.t_invert * input, dim=self.dim)

        shape = list(input.shape)
        if self.dim[0] != 0:
            shape_dim_0 = shape[self.dim[0]]
            input = input.permute(0, self.dim[0])
            shape[self.dim[0]] = shape[0]
            shape[0] = shape_dim_0
        if self.dim[1] != 1:
            shape_dim_1 = shape[self.dim[1]]
            input = input.permute(1, self.dim[1])
            shape[self.dim[1]] = shape[1]
            shape[1] = shape_dim_1

        input = input.view([shape[0]*shape[1]]+shape[2:])
        input = torch.softmax(self.t_invert * input, dim=0)
        input = input.view(shape)
        if self.dim[1] != 1:
            input = input.permute(1, self.dim[1])
        if self.dim[0] != 0:
            input = input.permute(0, self.dim[0])
        return input




    def extra_repr(self) -> str:
        return 't_invert=%s, dim=%s'%(self.t_invert, self.dim)