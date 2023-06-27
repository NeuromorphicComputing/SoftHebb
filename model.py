import torch
import torch.nn as nn

try:
    from utils import RESULT, activation
except:
    from hebb.utils import RESULT, activation
from layer import generate_block
import os
import os.path as op


def load_layers(params, model_name, resume=None, verbose=True, model_path_override=None):
    """
    Create Model and load state if resume
    """

    if resume is not None:
        if model_path_override is None:
            model_path = op.join(RESULT, 'network', model_name, 'models', 'checkpoint.pth.tar')
        else:
            model_path = model_path_override

        if op.isfile(model_path):
            checkpoint = torch.load(model_path)  # , map_location=device)
            state_dict = checkpoint['state_dict']
            params2 = checkpoint['config']
            if resume == 'without_classifier':
                classifier_key = list(params.keys())[-1]
                params2[classifier_key] = params[classifier_key]

            model = MultiLayer(params2)

            state_dict2 = model.state_dict()

            if resume == 'without_classifier':
                for key, value in state_dict.items():
                    if resume == 'without_classifier' and str(params[classifier_key]['num']) in key:
                        continue
                    if key in state_dict2:
                        state_dict2[key] = value
                model.load_state_dict(state_dict2)
            else:
                model.load_state_dict(state_dict)
            # log.from_dict(checkpoint['measures'])
            starting_epoch = 0  # checkpoint['epoch']
            print('\n', 'Model %s loaded successfuly with best perf' % (model_name))
            # shutil.rmtree(op.join(RESULT, params.folder_name, 'figures'))
            # os.mkdir(op.join(RESULT, params.folder_name, 'figures'))
        else:
            print('\n', 'Model %s not found' % model_name)
            model = MultiLayer(params)
        print('\n')
    else:
        model = MultiLayer(params)

    if verbose:
        model.__str__()

    return model


def save_layers(model, model_name, epoch, blocks, filename='checkpoint.pth.tar', storing_path=None):
    """
    Save model and each of its training blocks
    """
    if storing_path is None:
        if not op.isdir(RESULT):
            os.makedirs(RESULT)
        if not op.isdir(op.join(RESULT, 'network')):
            os.mkdir(op.join(RESULT, 'network'))
            os.mkdir(op.join(RESULT, 'layer'))

        folder_path = op.join(RESULT, 'network', model_name)
        if not op.isdir(folder_path):
            os.makedirs(op.join(folder_path, 'models'))
        storing_path = op.join(folder_path, 'models')


    torch.save({
        'state_dict': model.state_dict(),
        'config': model.config,
        'epoch': epoch
    }, op.join(storing_path, filename))

    for block_id in blocks:
        block = model.get_block(block_id)
        block_path = op.join(RESULT, 'layer', 'block%s' % block.num)
        if not op.isdir(block_path):
            os.makedirs(block_path)
        folder_path = op.join(block_path, block.get_name())
        if not op.isdir(folder_path):
            os.mkdir(folder_path)
        torch.save({
            'state_dict': block.state_dict(),
            'epoch': epoch
        }, op.join(folder_path, filename))


class MultiLayer(nn.Module):
    """
       MultiLayer Network created from list of preset blocks
    """

    def __init__(self, blocks_params: dict, blocks: nn.Module = None) -> None:
        super().__init__()
        self.train_mode = None
        self.train_blocks = []

        self.config = blocks_params
        if blocks_params is not None:
            blocks = []
            for _, params in blocks_params.items():
                blocks.append(generate_block(params))
            self.blocks = nn.Sequential(*blocks)
        else:
            self.blocks = nn.Sequential(*blocks)

    def foward_x_wta(self, x: torch.Tensor) -> torch.Tensor:
        for id, block in self.generator_block():
            if id != len(self.blocks) - 1:
                x = block(x)
            else:
                return block.foward_x_wta(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x

    def get_block(self, id):
        return self.blocks[id]

    def sub_model(self, block_ids):
        sub_blocks = []
        max_id = max(block_ids)
        for id, block in self.generator_block():
            sub_blocks.append(self.get_block(id))
            if id == max_id:
                break

        return MultiLayer(None, sub_blocks)

    def is_hebbian(self) -> bool:
        """
        Return if the last block of the model is hebbian
        """
        return self.blocks[-1].is_hebbian()

    def get_lr(self) -> float:
        """
        Return the lr of the last hebbian block
        """
        if self.train_blocks:
            for i in reversed(self.train_blocks):
                if self.blocks[-i].is_hebbian():
                    return self.blocks[-i].get_lr()
        if self.blocks[0].is_hebbian():
            return self.blocks[0].get_lr()
        return 0

    def radius(self, layer=None) -> str:
        """
        Return the radius of the first hebbian block
        """
        if layer is not None:
            return self.blocks[layer].radius()
        if self.train_blocks:
            r = []
            for i in reversed(self.train_blocks):
                if self.blocks[i].is_hebbian():
                    r.append(self.blocks[i].radius())
            return '\n ************************************************************** \n'.join(r)
        if self.blocks[0].is_hebbian():
            return self.blocks[0].radius()
        return ''

    def convergence(self) -> str:
        """
        Return the radius of the last hebbian block
        """
        for i in range(1, len(self.blocks) + 1):
            if self.blocks[-i].is_hebbian():
                return self.blocks[-i].layer.convergence()
        return 0, 0

    def reset(self):
        if self.blocks[0].is_hebbian():
            self.blocks[0].layer.reset()

    def generator_block(self):
        for id, block in enumerate(self.blocks):
            yield id, block

    def update(self):
        for block in self.train_blocks:
            self.get_block(block).update()

    def __str__(self):
        for _, block in self.generator_block():
            block.__str__()

    def train(self, mode=True, blocks=[]):
        """
        Set the learning update to the expected mode.
        mode:True, BP:False, HB:True --> training Hebbian layer
        mode:True, BP:True, HB:False --> training fc
        mode:True, BP:True, HB:True --> training Hebbain + fc blocks
        mode:False --> predict
        """
        self.training = mode
        self.train_blocks = blocks
        # print('train mode %s and layer %s'%(mode, blocks))

        for param in self.parameters():
            param.requires_grad = False
        for _, block in self.generator_block():
            block.eval()

        for block in blocks:
            module = self.get_block(block)

            module.train(mode)
            for param in module.parameters():
                param.requires_grad = True


class HebbianOptimizer:
    def __init__(self, model):
        """Custom optimizer which particularly delegates weight updates of Unsupervised layers to these layers themselves.

        Args:
            model (torch.nn.Module): Pytorch model
        """
        self.model = model
        self.param_groups = []

    @torch.no_grad()
    def step(self, *args):
        """Performs a single optimization step.
        """
        loss = None

        for block in self.model.blocks:
            if block.is_hebbian():
                block.update(*args)

    def zero_grad(self):
        pass


class AggregateOptim:
    def __init__(self, optimizers):
        """Custom optimizer aggregating several optimizers together to run simulaneously

        Args:
            optimizers (List[torch.autograd.optim.Optimizer]): List of optimizers which need to be called simultaneously
        """
        self.optimizers = optimizers
        self.param_groups = []
        for optim in self.optimizers:
            self.param_groups.extend(optim.param_groups)

    def __repr__(self):
        representations = []
        for optim in self.optimizers:
            representations.append(repr(optim))
        return '\n'.join(representations)

    def step(self):
        for optim in self.optimizers:
            optim.step()

    def zero_grad(self):
        for optim in self.optimizers:
            optim.zero_grad()
