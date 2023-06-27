from utils import CustomStepLR, double_factorial
from model import save_layers, HebbianOptimizer, AggregateOptim
from engine import train_sup, train_unsup, evaluate_unsup, evaluate_sup
from dataset import make_data_loaders
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np


def check_dimension(blocks, dataset_config):
    """
    Make each block dimension of the model corresponds to the next one.
    Parameters
    ----------
    blocks: dict
        configuration of every blocks in the model
    dataset_config: dict
        configuration of the dataset
    Returns
    -------
        blocks: dict
           configuration of every blocks in the model with correct dimensionality

    """

    in_channels, out_channels_final, in_width, in_height = dataset_config['channels'], \
                                                           dataset_config['out_channels'], \
                                                           dataset_config['width'], \
                                                           dataset_config['height']

    for id in range(len(blocks)):
        block = blocks['b%s' % id]
        assert block['num'] == id, 'Block b%s has not the correct number %s ' % (id, block['num'])
        config = block['layer']
        if id == len(blocks) - 1 and not config['hebbian']:
            config['out_channels'] = out_channels_final
            # assert out_channels_final == config['out_channels'], \
            #   'Output channels %s is different than number of classes %s'%(config['out_channels'], out_channels_final)

        if 'operation' in block and 'flatten' in block['operation']:
            config['in_channels'] = int(in_channels * in_width * in_height)
            config['old_channels'] = in_channels
        else:
            config['in_channels'] = in_channels

        if block['arch'] == 'CNN':
            # config['padding'] = config['kernel_size']//2
            in_width = int((in_width + 2 * config['padding'] - config['dilation'] * (
                    config['kernel_size'] - 1) - 1) / config['stride']) + 1
            in_height = int((in_height + 2 * config['padding'] - config['dilation'] * (
                    config['kernel_size'] - 1) - 1) / config['stride']) + 1
            if block['pool'] is not None:
                # block['pool']['padding'] = int(int(block['pool']['kernel_size']) / 2 - 1)
                in_width = int((in_width - 1 * (block['pool']['kernel_size'] - 1) + 2 * block['pool']['padding'] - 1) /
                               block['pool']['stride'] + 1)
                in_height = int(
                    (in_height - 1 * (block['pool']['kernel_size'] - 1) + 2 * block['pool']['padding'] - 1) /
                    block['pool']['stride'] + 1)
            print('block %s, size : %s %s %s' % (id, config['out_channels'], in_width, in_height))
        in_channels = config['out_channels']  # prepare for next loop

        lp = blocks['b%s' % id]['layer']['lebesgue_p']
        initial_r = blocks['b%s' % id]['layer']['radius'] ** (lp)

        if blocks['b%s' % id]['arch'] == 'CNN':
            kenel_size = blocks['b%s' % id]['layer']['kernel_size']
            input_channel = blocks['b%s' % id]['layer']['in_channels']
            groups = blocks['b%s' % id]['layer']['groups']
            n_neurons = input_channel / groups * kenel_size ** 2
        else:
            n_neurons = blocks['b%s' % id]['layer']['in_channels']
        if "operation" in block and "batchnorm" in block["operation"]:
            blocks['b%s' % id]['layer']['weight_init'] = 'normal'

            t = double_factorial(lp - 1) * (np.sqrt(2 / np.pi) if lp % 2 != 0 else 1)
            blocks['b%s' % id]['layer']['weight_init_range'] = np.power((initial_r / (n_neurons * t)), 1 / lp)
        else:
            blocks['b%s' % id]['layer']['weight_init'] = 'positive'
            blocks['b%s' % id]['layer']['weight_init_range'] = np.power(((lp + 1) * initial_r / (n_neurons)), 1 / lp)

        print('range = %s' % blocks['b%s' % id]['layer']['weight_init_range'])
    return blocks


def training_config(blocks, dataset_sup_config, dataset_unsup_config, mode, blocks_train=None):
    """
    Define the training order of blocks:
    -successive: one block after the other
    -consecutive: Hebbian blocks then BP blocks
    -simultaneous: All at once with an hybrid learning

    Parameters
    ----------
    blocks: dict
        configuration of every blocks in the model
    dataset_config: dict
        configuration of the dataset
    mode: str

    Returns
    -------
        train_layer_order: dict
            configuration of the training blocks order


    """
    for id in range(len(blocks)):
        blocks['b%s' % id]['layer']['lr_scheduler'] = {'decay': 'cste', 'lr': 0.1}
    blocks_train = range(len(blocks)) if blocks_train is None else blocks_train
    if mode == 'successive':
        train_layer_order = {}
        train_id = 0
        for id in blocks_train:
            block = blocks['b%s' % id]
            config = block['layer']
            if config['hebbian']:

                train_layer_order['t%s' % train_id] = {
                    'blocks': [id],
                    'mode': 'unsupervised',
                    'lr': config['lr'],
                    'nb_epoch': dataset_unsup_config['nb_epoch'],
                    'batch_size': dataset_unsup_config['batch_size'],
                    'print_freq': dataset_unsup_config['print_freq']
                }
                config['lr_scheduler'] = {
                    'lr': config['lr'],
                    'adaptive': config['adaptive'],
                    'nb_epochs': dataset_unsup_config['nb_epoch'],
                    'ratio': dataset_unsup_config['batch_size'] / dataset_unsup_config['training_sample'],
                    'speed': config['speed'],
                    'div': config['lr_div'],
                    'decay': config['lr_decay'],
                    'power_lr': config['power_lr']
                }
                last_hebbian = True
                train_id += 1
            else:
                train_layer_order['t%s' % train_id] = {
                    'blocks': [id],
                    'mode': 'supervised',
                    'lr': config['lr_sup'],
                    'nb_epoch': dataset_sup_config['nb_epoch'],
                    'batch_size': dataset_sup_config['batch_size'],
                    'print_freq': dataset_sup_config['print_freq']
                }
            train_id += 1
    elif mode == 'consecutive':
        train_layer_order = {}
        layer = {'sup': [], 'unsup': []}
        lr = {'sup': [], 'unsup': []}

        for id in blocks_train:
            block = blocks['b%s' % id]
            config = block['layer']
            # this allows to have supervised Hebbian
            is_unsup = config['hebbian'] and config.get('metric_mode', 'unsupervised') != 'supervised'
            if is_unsup:
                layer['unsup'].append(id)
                lr['unsup'].append(config['lr'])
            else:
                layer['sup'].append(id)
                lr['sup'].append(config['lr_sup'])
        if layer['unsup']:  # if the list is not empty, i.e. we have unsup blocks
            train_layer_order['t0'] = {
                'blocks': layer['unsup'],
                'mode': 'unsupervised',
                'batch_size': dataset_unsup_config['batch_size'],
                'nb_epoch': dataset_unsup_config['nb_epoch'],
                'print_freq': dataset_unsup_config['print_freq'],
                'lr': min(lr['unsup'])
            }
        if layer['sup']:  # if the list is not empty, i.e. we have sup blocks
            t_id = 't1' if layer['unsup'] else 't0'
            train_layer_order[t_id] = {
                'blocks': layer['sup'],
                'mode': 'supervised',
                'batch_size': dataset_sup_config['batch_size'],
                'nb_epoch': dataset_sup_config['nb_epoch'],
                'print_freq': dataset_sup_config['print_freq'],
                'lr': min(lr['sup'])
            }
        for id in range(len(blocks)):
            block = blocks['b%s' % id]
            config = block['layer']
            if config['hebbian']:
                config['lr_scheduler'] = {
                    'lr': config['lr'],
                    'adaptive': config['adaptive'],
                    'nb_epochs': train_layer_order['t0']['nb_epoch'],
                    'ratio': train_layer_order['t0']['batch_size'] / dataset_unsup_config['training_sample'],
                    'speed': config['speed'],
                    'div': config['lr_div'],
                    'decay': config['lr_decay'],
                    'power_lr': config['power_lr']
                }
    elif mode == 'simultaneous':
        train_layer_order = {
            'blocks': [],
            'lr': [],
            'mode': 'hybrid',
            'batch_size': dataset_sup_config['batch_size'],
            'nb_epoch': dataset_sup_config['nb_epoch'],
            'print_freq': dataset_sup_config['print_freq'],
        }

        for id in blocks_train:
            block = blocks['b%s' % id]
            config = block['layer']
            train_layer_order['blocks'].append(id)
            if not config['hebbian']:
                train_layer_order['lr'].append(config['lr_sup'])

        train_layer_order['lr'] = min(train_layer_order['lr'])
        for id in range(len(blocks)):
            block = blocks['b%s' % id]
            config = block['layer']
            if config['hebbian']:
                config['lr_scheduler'] = {
                    'lr': config['lr'],
                    'adaptive': config['adaptive'],
                    'nb_epochs': train_layer_order['nb_epoch'],
                    'ratio': train_layer_order['batch_size'] / dataset_sup_config['training_sample'],
                    'speed': config['speed'],
                    'div': config['lr_div'],
                    'decay': config['lr_decay'],
                    'power_lr': config['power_lr']
                }
        train_layer_order = {'t1': train_layer_order}
    else:
        raise ValueError
    return train_layer_order


def run_hybrid(
        final_epoch: int,
        print_freq: int,
        batch_size: int,
        lr: float,
        folder_name: str,
        dataset_config: dict,
        model,
        device,
        log,
        blocks,
        learning_mode: str = 'BP',
        save_batch: bool = True,
        save: bool = True,
        report=None,
        plot_fc=None,
        model_dir=None,
):
    """
    Hybrid training of one model, happens during simultaneous training mode

    """

    print('\n', '********** Hybrid learning of blocks %s **********' % blocks)

    train_loader, test_loader = make_data_loaders(dataset_config, batch_size, device)

    optimizer_sgd = optim.Adam(
        model.parameters(), lr=lr)  # , weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    hebbian_optimizer = HebbianOptimizer(model)
    scheduler = CustomStepLR(optimizer_sgd, final_epoch)
    optimizer = AggregateOptim((hebbian_optimizer, optimizer_sgd))
    log_batch = log.new_log_batch()
    for epoch in range(1, final_epoch + 1):
        measures, lr = train_sup(model, criterion, optimizer, train_loader, device, log_batch, learning_mode, blocks)

        if scheduler is not None:
            scheduler.step()

        if epoch % print_freq == 0 or epoch == final_epoch or epoch == 1:

            loss_test, acc_test = evaluate_sup(model, criterion, test_loader, device)

            log_batch = log.step(epoch, log_batch, loss_test, acc_test, lr, save=save_batch)

            if report is not None:
                _, train_loss, train_acc, test_loss, test_acc = log.data[-1]

                conv, R1 = model.convergence()
                report(train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
                       convergence=conv, R1=R1)

            else:
                log.verbose()
            if save:
                save_layers(model, folder_name, epoch, blocks, storing_path=model_dir)

            if plot_fc is not None:
                for block in blocks:
                    plot_fc(model, block)


def run_unsup(
        final_epoch: int,
        print_freq: int,
        batch_size: int,
        folder_name: str,
        dataset_config: dict,
        model,
        device,
        log,
        blocks,
        save: bool = True,
        report=None,
        plot_fc=None,
        reset=False,
        model_dir=None
):
    """
    Unsupervised training of hebbians blocks of one model

    """
    print('\n', '********** Hebbian Unsupervised learning of blocks %s **********' % blocks)

    train_loader, test_loader = make_data_loaders(dataset_config, batch_size, device)

    for epoch in range(1, final_epoch + 1):
        lr, info, convergence, R1 = train_unsup(model, train_loader, device, blocks)

        if epoch % print_freq == 0 or epoch == final_epoch or epoch == 1:

            acc_train, acc_test = evaluate_unsup(model, train_loader, test_loader, device, blocks)

            log.step(epoch, acc_train, acc_test, info, convergence, R1, lr)

            if report is not None:
                report(train_loss=0., train_acc=acc_train, test_loss=0., test_acc=acc_test, convergence=convergence,
                       R1=R1)
            # else:
            log.verbose()

            if save:
                save_layers(model, folder_name, epoch, blocks, storing_path=model_dir)

            if plot_fc is not None:
                for block in blocks:
                    plot_fc(model, block)
    if reset:
        model.reset()


def run_sup(
        final_epoch: int,
        print_freq: int,
        batch_size: int,
        lr: float,
        folder_name: str,
        dataset_config: dict,
        model,
        device,
        log,
        blocks,
        learning_mode: str = 'BP',
        save_batch: bool = False,
        save: bool = True,
        report=None,
        plot_fc=None,
        model_dir=None
):
    """
    Supervised training of BP blocks of one model

    """

    print('\n', '********** Supervised learning of blocks %s **********' % blocks)

    train_loader, test_loader = make_data_loaders(dataset_config, batch_size, device)

    criterion = nn.CrossEntropyLoss()
    log_batch = log.new_log_batch()
    if all([model.get_block(b).is_hebbian() for b in blocks]):
        # optimizer, scheduler, log_batch = None, None, None
        optimizer, scheduler = None, None
    else:
        # criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)  # , weight_decay=1e-6)
        scheduler = CustomStepLR(optimizer, final_epoch)

    for epoch in range(1, final_epoch + 1):
        measures, lr = train_sup(model, criterion, optimizer, train_loader, device, log_batch, learning_mode, blocks)

        if scheduler is not None:
            scheduler.step()

        if epoch % print_freq == 0 or epoch == final_epoch or epoch == 1:

            # so the diff between evaluate sup and unsup is that former calcs train and test acc, former test loss and acc
            loss_test, acc_test = evaluate_sup(model, criterion, test_loader, device)

            log_batch = log.step(epoch, log_batch, loss_test, acc_test, lr, save_batch)

            if report is not None:
                _, train_loss, train_acc, test_loss, test_acc = log.data[-1]
                conv, R1 = model.convergence()
                report(train_loss=train_loss, train_acc=train_acc, test_loss=test_loss, test_acc=test_acc,
                       convergence=conv, R1=R1)
            else:
                log.verbose()

            if save:
                save_layers(model, folder_name, epoch, blocks, storing_path=model_dir)

            if plot_fc is not None:
                for block in blocks:
                    plot_fc(model, block)
