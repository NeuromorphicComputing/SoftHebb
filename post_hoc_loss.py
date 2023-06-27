import argparse

from utils import load_presets, get_device, load_config_dataset, seed_init_fn
from model import load_layers
from train import run_sup, check_dimension
from log import Log, save_logs
import copy
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Post hoc loss')

parser.add_argument('--preset', choices=load_presets(), default=None,
                    type=str, help='Preset of hyper-parameters ' +
                        ' | '.join(load_presets()) +
                        ' (default: None)')

parser.add_argument('--dataset-unsup', choices=load_config_dataset(), default='MNIST',
                    type=str, help='Dataset possibilities ' +
                        ' | '.join(load_config_dataset()) +
                        ' (default: MNIST)')

parser.add_argument('--dataset-sup', choices=load_config_dataset(), default='MNIST',
                    type=str, help='Dataset possibilities ' +
                        ' | '.join(load_config_dataset()) +
                        ' (default: MNIST)')

parser.add_argument('--model-name', default=None, type=str, help='Model Name')


parser.add_argument('--seed', default=None,  type=int,
                    help='Selection of the blocks that will be trained')

parser.add_argument('--gpu-id', default=0, type=int, metavar='N',
                    help='Id of gpu selected for training (default: 0)')


def training_config(blocks, dataset_sup_config, dataset_unsup_config):
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
        blocks['b%s' % id]['layer']['lr_scheduler'] = {'decay':'cste', 'lr': 0.1}
    blocks_train = range(len(blocks))

    train_layer_order = {}
    layer = {'sup':[], 'unsup':[]}
    lr = {'sup': [], 'unsup': []}

    for id in blocks_train:
        block = blocks['b%s' % id]
        config = block['layer']
        if config['hebbian']:
            layer['unsup'].append(id)
            lr['unsup'].append(config['lr'])

        else:
            layer['sup'].append(id)
            lr['unsup'].append(config['lr'])
            lr['sup'].append(config['lr_sup'])
    if layer['unsup']:
        train_layer_order['t0'] = {
            'blocks': layer['unsup'],
            'mode': 'supervised',
            'learning_mode': 'HB',
            'batch_size': dataset_unsup_config['batch_size'],
            'nb_epoch': dataset_unsup_config['nb_epoch'],
            'print_freq': dataset_unsup_config['print_freq'],
            'lr': min(lr['unsup'])
        }
        config =  blocks['b0']['layer']

        config['lr_scheduler'] = {
            'lr': config['lr'],
            'adaptive': config['adaptive'],
            'nb_epochs':  train_layer_order['t0']['nb_epoch'],
            'ratio': train_layer_order['t0']['batch_size'] / dataset_unsup_config['training_sample'],
            'speed': config['speed'],
            'div': config['lr_div'],
            'decay': config['lr_decay'],
            'power_lr': config['power_lr']
        }
    else:
        train_layer_order['t0'] = {
            'blocks': layer['sup'],
            'mode': 'supervised',
            'learning_mode': 'BP',
            'batch_size': dataset_unsup_config['batch_size'],
            'nb_epoch': dataset_unsup_config['nb_epoch'],
            'print_freq': dataset_unsup_config['print_freq'],
            'lr': min(lr['unsup'])
        }
    train_layer_order['t2'] = train_layer_order['t0'].copy()
    train_layer_order['t2']['blocks'] = [0]

    train_layer_order['t1'] = {
        'blocks': [1],
        'mode': 'supervised',
        'batch_size': dataset_sup_config['batch_size'],
        'nb_epoch': dataset_sup_config['nb_epoch'],
        'print_freq': dataset_sup_config['print_freq'],
        'lr': min(lr['sup'])
    }

    return train_layer_order


def main(blocks, name_model, dataset_sup_config, dataset_unsup_config, train_config, gpu_id):
    device = get_device(gpu_id)
    model = load_layers(blocks, name_model, resume=None)

    model = model.to(device)

    log = Log(train_config)

    # copy all the parameter of the first layer to overide in the future
    b0_copy = copy.deepcopy(model.get_block(0).state_dict())

    config0 = train_config['t0']
    config1 = train_config['t1']
    config2 = train_config['t2']

    if config0['learning_mode'] == 'HB':
        run_sup(
            final_epoch=config0['nb_epoch'],
            print_freq=config0['print_freq'],
            batch_size=config0['batch_size'],
            lr=config0['lr'],
            folder_name=name_model,
            dataset_config=dataset_unsup_config,
            model=model,
            device=device,
            log=log.sup['t0'],
            blocks=config0['blocks'],
            learning_mode=config0['learning_mode']
        )
        print('First layer trained')

    run_sup(
        final_epoch=config1['nb_epoch'],
        print_freq=config1['print_freq'],
        batch_size=config1['batch_size'],
        lr=config1['lr'],
        folder_name=name_model,
        dataset_config=dataset_sup_config,
        model=model,
        device=device,
        log=log.sup['t1'],
        blocks=config1['blocks']
    )


    print('Second layer trained')
    # overide parameters of the first layer

    print(model.get_block(0).state_dict())
    print(b0_copy)
    model.get_block(0).load_state_dict(b0_copy)
    model.reset()


    run_sup(
        final_epoch=config2['nb_epoch'],
        print_freq=config2['print_freq'],
        batch_size=config2['batch_size'],
        lr=config2['lr'],
        folder_name=name_model,
        dataset_config=dataset_sup_config,
        model=model,
        device=device,
        log=log.sup['t2'],
        blocks=config2['blocks'],
        learning_mode=config2['learning_mode'],
        save_batch=True
    )

    save_logs(log, name_model)


if __name__ == '__main__':
    params = parser.parse_args()
    name_model = params.preset if params.model_name is None else params.model_name
    blocks = load_presets(params.preset)
    dataset_unsup_config = load_config_dataset(params.dataset_unsup, False)
    dataset_sup_config = load_config_dataset(params.dataset_sup, False)
    if params.seed is not None:
        dataset_sup_config['seed'] = params.seed

    if dataset_sup_config['seed'] is not None:
        seed_init_fn(dataset_sup_config['seed'])

    blocks = check_dimension(blocks, dataset_sup_config)


    train_config = training_config(blocks, dataset_sup_config, dataset_unsup_config)

    main(blocks, name_model, dataset_sup_config, dataset_unsup_config, train_config, params.gpu_id)
