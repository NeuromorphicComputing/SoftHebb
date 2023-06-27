import argparse
import os.path as op
import json
import random
from math import e

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR

username = op.expanduser('~').split('/')[-1]
data_candidate = ('/scratch' if 'hrodriguez' == username else '/home') + f'/{username}/workspace'
DATA = op.realpath(op.expanduser(data_candidate))
RESULT = op.join(DATA, 'results', 'hebb', 'result')  # everything from multi_layer.py
SEARCH = op.join(DATA, 'results', 'hebb', 'search')  # everything from ray_search
DATASET = op.join(DATA, 'data')


def get_folder_name(params):
    """
    from a set of parameter, define the name of the model and thus its folder name
    

    Parameters
    ----------
    params : namespace or dict
        hyperparameters.

    Returns
    -------
    folder_name : str
        folder name or name of one model.

    """
    if params.folder_name is not None:
        return params.folder_name
    if params.preset is not None:
        folder_name = params.preset

    if isinstance(params, dict):
        if params['folder_name'] is not None:
            return params.folder_name
        if params['preset'] is not None:
            folder_name = params.preset
        else:
            names = ['arch', 'n_neurons', 'lr', 't_invert']
            folder_name = '_'.join([str(params[name]) for name in names])
        if params['post_hoc_loss']:
            folder_name = 'post_hoc_loss_' + folder_name
    else:
        if params.folder_name is not None:
            return params.folder_name
        if params.preset is not None:
            folder_name = params.preset
        else:
            names = ['arch', 'n_neurons', 'lr', 't_invert']
            folder_name = '_'.join([str(getattr(params, name)) for name in names])
        if params.post_hoc_loss:
            folder_name = 'post_hoc_loss_' + folder_name

    return folder_name


def activation(x, t_invert=e, activation_fn='exp', dim=1, power=15, beta=1, normalize=False):
    """
    Applies the Softmax function to an n-dimensional input Tensor
    rescaling them so that the elements of the n-dimensional output Tensor
    lie in the range [0,1] and sum to 1. It using softmax function from pytorch.
    The t_invert parameter allows a range of softmax between WTA and AllTA.

    Parameters
    ----------
    x : torch.tensor
        DESCRIPTION.
    t_invert : torch.tensor
        DESCRIPTION. The default is torch.tensor(e).
    activation_fn : str
        activation function name. The default is 'exp'.
    dim : int
        output dimension of the softmax. The default is 1.

    Returns
    -------
    TYPE
        softmax compute as a torch.tensor.

    """
    if (activation_fn == 'exp' and normalize) or activation_fn == 'softmax':
        # this can lead to erros when passed with -inf, which is a design choice of funcs that call this
        # it'd be good to use a custom softmax where we could pass a small value in the denominator
        # however it seems it is not trivial to construct a softmax that achieves similar performance as Pytorch's
        return torch.softmax(t_invert * x, dim)
    if activation_fn == 'exp':
        x = torch.exp(x * t_invert)
    elif activation_fn == 'relu':
        x = torch.relu(x)
    elif activation_fn == 'sigmoid':
        x = torch.sigmoid(x)
    elif activation_fn == 'repu':
        x = torch.relu(x) ** power
    elif activation_fn == 'repu_norm':
        x = torch.relu(x) ** power
        normalize = True
    elif activation_fn == 'tanh':
        x = torch.tanh(beta * x)
    if normalize and x.sum() != 0:
        return (x.t() / x.sum(dim=1)).t()
    else:
        return x


def get_device(gpu_id=0):
    """
    Get the correct device either cuda or cpu with the selected id.

    Parameters
    ----------
    gpu_id : int
        Gpu id. The default is 0.

    Returns
    -------
    device : torch.device
        torch device either gpu or cpu.

    """
    use_cuda = torch.cuda.is_available() and gpu_id is not None
    device = torch.device('cuda:' + str(gpu_id) if use_cuda else 'cpu')
    return device


def seed_init_fn(seed):
    """
    Dataloader worker init function, if seed is not None every epoch and 
    experiment will get the same data.

    Parameters
    ----------
    seed : int
        seed Id.

    Returns
    -------
    None.

    """
    seed = seed % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    return


def merge_parameter(based_params, override_params):
    """
    Update the parameters in ``t_invert_params`` with ``override_params``.
    Can be useful to override parsed command line arguments.

    
    Parameters
    ----------
    params : namespace or dict
        t_invert parameters. A key-value mapping.
    override_params : dict or None
        Parameters to override. Usually the parameters got from ``get_next_parameters()``.
        When it is none, nothing will happen.

    Returns
    -------
    params : namespace or dict
        The updated ``t_invert_params``. Note that ``t_invert_params`` will be updated inplace. The return value is
        only for convenience..

    """
    if override_params is None:
        return based_params
    is_dict = isinstance(based_params, dict)
    for k, v in override_params.items():
        if is_dict:
            # if k not in params:
            #    raise ValueError('Key \'%s\' not found in parameters.' % k)
            if k not in based_params:
                based_params[k] = v
            elif isinstance(based_params[k], dict):
                if isinstance(v, dict):
                    based_params[k] = merge_parameter(based_params[k], v)
            else:
                based_params[k] = v
        else:
            # if not hasattr(params, k):
            #    raise ValueError('Key \'%s\' not found in parameters.' % k)
            if not hasattr(based_params, k):
                setattr(based_params, k, v)
            elif isinstance(getattr(based_params, k), dict):
                if isinstance(v, dict):
                    setattr(based_params, k, merge_parameter(based_params[k], v))
            else:
                setattr(based_params, k, v)
    return based_params


def str2bool(v):
    """
    Return boolean form a string

    Parameters
    ----------
    v : str
        argparse argument.

    Returns
    -------
    bool
        DESCRIPTION.

    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_weight(shape, weight_distribution, weight_range, weight_offset=0):
    """
    Weight initialization from a distribution
    Parameters
    ----------
    shape: tuple
        Expected shape of the Weight tensor
    weight_distribution: str
        Distribution
    weight_range:
        multiplier of the weight
    weight_offset:
        Value add to the weight

    Returns
    -------
        weight: Tensor
    """
    if weight_distribution == 'positive':
        return weight_range * torch.rand(shape) + weight_offset
    elif weight_distribution == 'negative':
        return -weight_range * torch.rand(shape) + weight_offset
    elif weight_distribution == 'zero_mean':
        return 2 * torch.rand(shape) + weight_offset
    elif weight_distribution == 'normal':
        return weight_range * torch.randn(shape) + weight_offset


def double_factorial(x):
    if x <= 2:
        return x
    return x * double_factorial(x - 2)


def LrLinearDecay(lr, nb_epoch, ratio):
    """
    Linear decay Generator
    """
    delta = lr * ratio / nb_epoch
    while True:
        yield max(0., lr)
        lr = lr - delta


def LrExpDecay(lr, nb_epoch, ratio, lr_div=100, speed=10):
    """
    Exponential decay Generator
    """
    relative_speed = speed * ratio / nb_epoch
    # to guaranty that min value is indeed lr / lr_div
    min_lr = (lr / lr_div - lr * np.exp(-speed)) / (1 - np.exp(-speed))

    while True:
        yield lr
        lr = (lr - min_lr) / np.exp(relative_speed) + min_lr


def LrCste(lr):
    """
    Constant decay Generator
    """
    while True:
        yield lr


def unsup_lr_scheduler(lr, nb_epochs=1, ratio=1, speed=1, div=150, decay: str = 'linear'):
    """
    Selection of the lr scheduler, return a Generator

    """
    if nb_epochs == 0 or decay == 'constant':
        return LrCste(lr)
    if decay == 'linear':
        return LrLinearDecay(lr, nb_epochs, ratio)
    if decay == 'exp':
        return LrExpDecay(lr, nb_epochs, ratio, speed=speed, lr_div=div)
    return LrCste(lr)


def normalize(normalize_type):
    if normalize_type == 'norm':
        return lambda x: nn.functional.normalize(x)
    return lambda x: x


def generate_config(preset, arch):
    """
    Generate config from name of the layer
    Parameters
    ----------
    preset: dict
        initial config
    arch: str
        Architecture

    Returns
    -------
        config: dict
    """
    config = {}
    preset = preset.split("-")
    if preset[0] == 'BP':
        config['hebbian'] = False
    else:
        config['hebbian'] = True
        config['softness'] = preset[0]

    for param in preset[1:]:
        if param.startswith('c'):
            config['out_channels'] = int(param[1:])
        if param.startswith('lr'):
            config['lr'] = float(param[2:])
        if param.startswith('ls'):
            config['lr_sup'] = float(param[2:])
        if param.startswith('lb'):
            config['lebesgue_p'] = float(param[2:])
        if param.startswith('lp'):
            config['power_lr'] = float(param[2:])
        if param.startswith('t'):
            config['t_invert'] = float(param[1:])
        if param.startswith('b'):
            config['add_bias'] = bool(int(param[1:]))
        if param.startswith('a'):
            config['delta'] = float(param[1:])
        if param.startswith('r'):
            config['radius'] = float(param[1:])
        if param.startswith('v'):
            config['adaptive'] = bool(int(param[1:]))

    if arch == 'CNN':
        for param in preset[1:]:
            if param.startswith('c'):
                config['out_channels'] = int(param[1:])
            elif param.startswith('k'):
                config['kernel_size'] = int(param[1:])
            elif param.startswith('d'):
                config['dilation'] = int(param[1:])
            elif param.startswith('p'):
                config['padding'] = int(param[1:])
            elif param.startswith('s'):
                config['stride'] = int(param[1:])
            elif param.startswith('s'):
                config['stride'] = int(param[1:])
            elif param.startswith('m'):
                config['mask_thsd'] = float(param[1:])
            elif param.startswith('g'):
                config['groups'] = int(param[1:])
            elif param.startswith('e'):
                config['pre_triangle'] = bool(int(param[1:]))

    return config


def load_presets(name=None):
    """
    Load blocks config from name of the models

    """
    presets = json.load(open('presets.json'))
    if name is None:
        return list(presets['model'].keys())
    blocks = presets['model'][name]
    for id, block in blocks.items():
        if block['preset'] in presets['layer'][block['arch']]:
            over_config = presets['layer'][block['arch']][block['preset']].copy()
        else:
            over_config = generate_config(block['preset'], block['arch'])  # an option is to pass the supervision here

        if 'layer' in blocks[id]:
            # had to add this to override parameters from the default layer (eg 'metric_mode' in MLP) without causing larger changes
            over_config = merge_parameter(over_config, blocks[id]['layer'])
        blocks[id]['layer'] = merge_parameter(presets['layer'][block['arch']]['default'].copy(), over_config)

        if 'pool' in block and block['pool'] is not None:
            type_, kernel_size, stride, padding = block['pool'].split('_')
            blocks[id]['pool'] = {'type': type_, 'kernel_size': int(kernel_size), 'stride': int(stride),
                                  'padding': int(padding)}
        else:
            blocks[id]['pool'] = None

        if 'activation' in block and block['activation'] is not None:
            param = 1
            activation = block['activation']
            activation_param = activation.split('_')
            if len(activation_param) == 2:
                activation = activation_param[0]
                param = float(activation_param[1])
            blocks[id]['activation'] = {'function': activation, 'param': param}
        else:
            blocks[id]['activation'] = None

    return blocks


def load_config_dataset(name=None, validation=True):
    """
    Load dataset config from name of the dataset

    """
    dataset = json.load(open('presets.json'))['dataset']
    if name is None:
        lst_dataset = []
        for key, value in dataset.items():
            for prop in value.keys():
                if prop == 'default':
                    lst_dataset.append(key)
                else:
                    lst_dataset.append(key + '_' + prop)

        return lst_dataset

    if '_' in name:
        dataset_name, dataset_prop = name.split('_')
    else:
        dataset_name = name
        dataset_prop = 'default'

    all_dataset_config = dataset[dataset_name]
    dataset_config = merge_parameter(dataset['default'], all_dataset_config['default'])
    dataset_config = merge_parameter(dataset_config, all_dataset_config[dataset_prop])

    dataset_config['validation'] = validation
    if validation:
        dataset_config['val_sample'] = int(
            np.floor(dataset_config['training_sample'] * dataset_config['validation_split']))
        dataset_config['training_sample'] = dataset_config['training_sample'] - dataset_config['val_sample']
    return dataset_config


class CustomStepLR(StepLR):
    def __init__(self, optimizer, nb_epochs):
        self.step_thresold = []
        if nb_epochs < 20:
            self.step_thresold = []
        elif nb_epochs < 50:
            self.step_thresold.append(int(nb_epochs * 0.5))
            self.step_thresold.append(int(nb_epochs * 0.75))
        else:
            self.step_thresold.append(int(nb_epochs * 0.2))
            self.step_thresold.append(int(nb_epochs * 0.35))
            self.step_thresold.append(int(nb_epochs * 0.5))
            self.step_thresold.append(int(nb_epochs * 0.6))
            self.step_thresold.append(int(nb_epochs * 0.7))
            self.step_thresold.append(int(nb_epochs * 0.8))
            self.step_thresold.append(int(nb_epochs * 0.9))

        super().__init__(optimizer, -1, False)

    def get_lr(self):
        if self.last_epoch in self.step_thresold:
            return [group['lr'] * 0.5
                    for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]


class PowerLoss(nn.Module):
    def __init__(self, nb_output=10, m=6):
        super().__init__()
        self.nb_output = nb_output
        self.m = m

    def forward(self, c, t):
        t = torch.eye(self.nb_output, dtype=torch.float, device=c.device)[t]
        t[t == 0] = -1.
        loss = (c - t).abs() ** self.m
        return loss.sum()
