import torch
from torch import Tensor
import torch.nn as nn

try:
    from utils import RESULT
except:
    from hebb.utils import RESULT

import torch.nn.functional as F
from typing import Callable, List, Optional
from hebblinear import select_linear_layer
from hebbconv import select_Conv2d_layer
from activation import get_activation
import os.path as op
import einops


class AttDropout(nn.Dropout):
    def forward(self, input):
        if self.training:
            nb_channels = input.shape[1]
            std = input.std(1)
            pb = self.p * ((-std) / std.max() + 1)
            pb = einops.repeat(pb, 'b w h -> b k w h', k=nb_channels)
            dropout = torch.bernoulli(pb)
            input[dropout == 1] = 0
            return input
        else:
            return input


class MaxNorm(nn.Module):
    def __init__(self, ) -> None:
        super(MaxNorm, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        shape = input.shape
        input_flat = input.reshape(shape[0], -1)

        return (input_flat / torch.unsqueeze(input_flat.amax(dim=1), 1)).view(shape)


class NormNorm(nn.Module):
    def __init__(self) -> None:
        super(NormNorm, self).__init__()
        self.multiplier = 10

    def forward(self, input: Tensor) -> Tensor:
        return self.multiplier * nn.functional.normalize(input)

    def extra_repr(self) -> str:
        return 'multiplier={}'.format(
            self.multiplier
        )


'''
class batchstd1d(nn.Module):
    def __init__(self, num_features) -> None:
        super(NormNorm, self).__init__()
        self.num_features = num_features
        self.register_buffer('running_std', torch.ones(num_features))
        self.register_buffer("num_batches_tracked", None)

    def forward(self, input: Tensor) -> Tensor:
        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1
            self.running_std = 

        return self.multiplier*nn.functional.normalize(input)

    def extra_repr(self) -> str:
        return 'multiplier={}'.format(
            self.multiplier
        )
'''


class BasicBlock(nn.Module):
    def __init__(
            self,
            arch: str,
            preset: str,
            num: int,
            in_channels: int,
            hebbian: bool,
            layer: nn.Module,
            resume: str = None,
            activation: Callable = None,
            operations: List = None,
            pool: Optional[nn.Module] = None,
            batch_norm: Optional[nn.Module] = None,
            dropout: Optional[nn.Module] = None,
            att_dropout: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        self.arch = arch
        self.preset = preset
        self.num = num
        self.in_channels = in_channels
        self.operations = operations
        self.layer = layer
        self.pool = pool
        self.batch_norm = batch_norm
        self.activation = activation
        self.dropout = dropout
        self.att_dropout = att_dropout
        self.hebbian = hebbian
        self.resume = resume
        if resume is not None:
            self.resume_block()

    def get_name(self):
        s = ''
        for operation in self.operations:
            s += operation.__class__.__name__
        if self.dropout is not None:
            s += self.dropout.__str__()
        if self.is_hebbian():
            s += self.layer.__label__()
        else:
            s += self.layer.__str__()[:10]
        if self.batch_norm is not None:
            s += self.batch_norm.__class__.__name__
        return s

    def is_hebbian(self):
        return self.hebbian

    def radius(self):
        return self.layer.radius()

    def get_lr(self):
        return self.layer.get_lr()

    def foward_x_wta(self, x: Tensor) -> Tensor:
        x = self.operations(x)
        return self.layer(x, return_x_wta=True)

    def update(self):
        if self.is_hebbian():
            self.layer.update()

    def sequential(self):
        elements = []
        if self.att_dropout is not None:
            elements.append(self.att_dropout)

        if self.operations:
            elements.append(self.operations)

        if self.dropout is not None:
            elements.append(self.dropout)

        elements.append(self.layer)

        if self.activation is not None:
            elements.append(self.activation)

        if self.batch_norm is not None:
            elements.append(self.batch_norm)

        if self.pool is not None:
            elements.append(self.pool)

        return nn.Sequential(*elements)

    def forward(self, x: Tensor) -> Tensor:
        # print('*'*(self.num+1), x.mean())
        # torch.cuda.empty_cache()

        if self.att_dropout is not None:
            x = self.att_dropout(x)

        x = self.operations(x)

        if self.dropout is not None:
            x = self.dropout(x)

        # x = self.layer(x.detach()) if self.is_hebbian()  else self.layer(x)
        x = self.layer(x)

        if self.activation is not None:
            x = self.activation(x)

        if self.batch_norm is not None:
            x = self.batch_norm(x)

        if self.pool is not None:
            x = self.pool(x)

        # torch.cuda.empty_cache()
        return x

    def __str__(self):
        print('\n', '----- Architecture Block %s, number %s -----' % (self.get_name(), self.num))
        if self.att_dropout is not None:
            print('-', self.att_dropout.__str__())
        for operation in self.operations:
            print('-', operation.__str__())
        if self.dropout is not None:
            print('-', self.dropout.__str__())
        print('-', self.layer.__str__())

        if self.activation is not None:
            print('-', self.activation.__str__())
        if self.batch_norm is not None:
            print('-', self.batch_norm.__str__())
        if self.pool is not None:
            print('-', self.pool.__str__())
        if self.resume is not None:
            print('***', self.resume)

    def resume_block(self, device: str = 'cpu'):
        model_path = op.join(RESULT, 'layer', 'block%s' % self.num, self.get_name(), 'checkpoint.pth.tar')
        if op.isfile(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=device)
                self.load_state_dict(checkpoint['state_dict'])
                self.resume = 'Block %s loaded successfuly' % self.get_name()
            except Exception as e:
                self.resume = 'File %s exist but %s' % (self.get_name(), e)
        else:
            self.resume = 'Block %s not found' % self.get_name()


def generate_block(params) -> BasicBlock:
    """

    Parameters
    ----------
    params

    Returns
    -------

    """
    config = params['layer']

    pool = None
    batch_norm = None
    operations = []
    activation = None
    dropout = None
    att_dropout = None

    if 'operation' in params:
        if 'batchnorm2d' in params['operation']:
            if config['arch'] == 'MLP':
                if 'flatten' in params['operation']:
                    operations.append(nn.BatchNorm2d(config['old_channels'], affine=False))
                else:
                    operations.append(nn.BatchNorm1d(config['in_channels'], affine=False))
            else:
                operations.append(nn.BatchNorm2d(config['in_channels'], affine=False))

        if 'flatten' in params['operation']:
            operations.append(nn.Flatten())

        if 'batchnorm1d' in params['operation']:
            if config['arch'] == 'MLP':
                operations.append(nn.BatchNorm1d(config['in_channels'], affine=False))
            elif 'flatten' in params['operation']:
                operations.append(nn.BatchNorm1d(config['in_channels'], affine=False))

        if 'max' in params['operation']:
            operations.append(MaxNorm())
        if 'normnorm' in params['operation']:
            operations.append(NormNorm())

    if config['arch'] == 'MLP':

        if config['hebbian']:
            layer = select_linear_layer(config)
        else:
            layer = nn.Linear(config['in_channels'], config['out_channels'])
        if 'batch_norm' in params and params['batch_norm']:
            batch_norm = nn.BatchNorm1d(config['out_channels'], affine=False)

    elif config['arch'] == 'CNN':
        if config['hebbian']:
            layer = select_Conv2d_layer(config)
        else:
            layer = nn.Conv2d(
                config['in_channels'],
                config['out_channels'],
                bias=config['add_bias'],
                kernel_size=config['kernel_size'],
                stride=config['stride'],
                padding=config['padding'],
                padding_mode=config['padding_mode'],
                dilation=config['dilation'],
                groups=config['groups']
            )
        if params['pool'] is not None:
            if params['pool']['type'] == 'max':
                pool = nn.MaxPool2d(kernel_size=params['pool']['kernel_size'], stride=params['pool']['stride'],
                                    padding=params['pool']['padding'])
            elif params['pool']['type'] == 'avg':
                pool = nn.AvgPool2d(kernel_size=params['pool']['kernel_size'], stride=params['pool']['stride'],
                                    padding=params['pool']['padding'])

        if 'batch_norm' in params and params['batch_norm']:
            batch_norm = nn.BatchNorm2d(config['out_channels'], affine=False)

    if params['activation'] is not None:
        activation = get_activation(
            activation_fn=params['activation']['function'],
            param=params['activation']['param'],
            dim=1)

    if 'dropout' in params and isinstance(params['dropout'], float):
        dropout = nn.Dropout(p=params['dropout'])

    if 'att_dropout' in params and isinstance(params['att_dropout'], float):
        att_dropout = AttDropout(p=params['att_dropout'])

    block = BasicBlock(
        arch=params['arch'],
        preset=params['preset'],
        num=params['num'],
        in_channels=config['in_channels'],
        hebbian=config['hebbian'],
        layer=layer,
        resume=None if "resume" not in params else params['resume'],
        activation=activation,
        operations=nn.Sequential(*operations),
        pool=pool,
        batch_norm=batch_norm,
        dropout=dropout,
        att_dropout=att_dropout
    )
    return block
