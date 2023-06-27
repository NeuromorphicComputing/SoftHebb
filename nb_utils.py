import csv
import json
import os
import os.path as op
from typing import Union, Any

import eagerpy as ep
import foolbox
from foolbox import PyTorchModel
from foolbox.attacks.base import T, get_criterion, raise_if_kwargs
from foolbox.attacks.gradient_descent_base import BaseGradientDescent, normalize_lp_norms, uniform_l2_n_balls
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.devutils import flatten
from foolbox.distances import l2
from foolbox.models.base import Model
from foolbox.types import Bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import umap

from model import load_layers
from log import load_logs
from utils import SEARCH


def find_checkpoint(path):
    for exp in os.listdir(path):
        exp_path = os.path.join(path, exp)
        if os.path.isdir(exp_path) and os.path.isfile(os.path.join(exp_path, 'checkpoint.pth.tar')):
            return exp_path
    print('No checkpoint found')
    return None


def resume_model(preset, model_path_override=None):
    model = load_layers([], preset, 'last', verbose=False, model_path_override=model_path_override)
    return model

def get_results(path, metric='test_acc'):
    seeds_results = []
    for exp in os.listdir(path):
        exp_path = os.path.join(path, exp)
        if os.path.isdir(exp_path):
            with open(os.path.join(exp_path, 'progress.csv')) as f:
                results = csv.DictReader(f)
                for row in results:
                    result = float(row[metric])
            seeds_results.append(result)
    assert len(seeds_results) > 0, 'Experiment results not found'
    return seeds_results

def get_mean_std(path, metric='test_acc'):
    seeds_results = get_results(path, metric)
    return np.mean(seeds_results), np.std(seeds_results)


class ChooseNeuronFlat(torch.nn.Module):
    def __init__(self, idx):
        super(ChooseNeuronFlat, self).__init__()
        self.idx = idx

    def forward(self, x):
        return x[:, self.idx].view(-1)


class SingleMax(foolbox.criteria.Criterion):
    def __init__(self, max_val: float, eps: float):
        super().__init__()
        self.max_val: float = max_val
        self.eps: float = eps

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(Max={self.max_val}, eps={self.eps})"

    def __call__(self, perturbed: T, outputs: T) -> T:
        outputs_, restore_type = ep.astensor_(outputs)
        del perturbed, outputs
        if self.max_val is None:
            is_adv = outputs_ > 0 or outputs_ <= 0
        else:
            is_adv = (self.max_val - outputs_) < self.eps
        return restore_type(is_adv)


def attack_model(model, data, labels, epsilons, iterations, random_start=True, nb_start=20, step_size=0.01 / 0.3,
                 criterion=None, device=None):
    attack = L2ProjGradientDescent(steps=iterations, random_start=random_start, rel_stepsize=step_size)
    bounds = (0, 1)
    fmodel = PyTorchModel(model, device=device, bounds=bounds)
    raw_advs, clipped_advs, success = attack(model=fmodel, inputs=data, criterion=criterion, epsilons=epsilons)
    raw_advs = torch.stack(raw_advs)
    clipped_advs = torch.stack(clipped_advs)
    # For debugging
    # print('Raw : x norm', raw_advs.norm().item(), 'x max', raw_advs.max().item(), 'x min', raw_advs.min().item())
    # print('Clipped : x norm', clipped_advs.norm().item(), 'x max', clipped_advs.max().item(), 'x min', clipped_advs.min().item())
    for start in range(1, nb_start):
        new_raw_advs, new_clipped_advs, new_success = attack(model=fmodel, inputs=data, criterion=criterion,
                                                             epsilons=epsilons)
        new_raw_advs = torch.stack(new_raw_advs)
        raw_advs[new_success == 1] = new_raw_advs[new_success == 1]

        new_clipped_advs = torch.stack(new_clipped_advs)
        clipped_advs[new_success == 1] = new_clipped_advs[new_success == 1]
        success[new_success == 1] = 1
    return raw_advs, clipped_advs, success


class L2ProjGradientDescent(BaseGradientDescent):
    distance = l2

    def get_random_start(self, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        print('Getting random start')
        batch_size, n = flatten(x0).shape
        r = uniform_l2_n_balls(x0, batch_size, n).reshape(x0.shape)
        return x0 + 0.00001 * epsilon * r

    def normalize(
            self, gradients: ep.Tensor, *, x: ep.Tensor, bounds: Bounds
    ) -> ep.Tensor:
        # This is to normalize gradients
        return gradients

    def project(self, x: ep.Tensor, x0: ep.Tensor, epsilon: float) -> ep.Tensor:
        sphere = epsilon * normalize_lp_norms(x, p=2).abs()
        sphere = sphere - sphere.min()
        if not sphere.max() > 0 and not sphere.min() < 0:
            return sphere
        else:
            return sphere / sphere.max()

    def run(
            self,
            model: Model,
            inputs: T,
            criterion: Union[Misclassification, TargetedMisclassification, T],
            *,
            epsilon: float,
            **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x0, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        del inputs, criterion, kwargs

        # perform a gradient ascent (targeted attack) or descent (untargeted attack)
        if isinstance(criterion_, Misclassification):
            gradient_step_sign = 1.0
            classes = criterion_.labels
            loss_fn = self.get_loss_fn(model, classes)
        elif hasattr(criterion_, "target_classes"):
            gradient_step_sign = -1.0
            classes = criterion_.target_classes  # type: ignore
            loss_fn = self.get_loss_fn(model, classes)
        elif hasattr(criterion_, "max_val"):
            def loss_fn_max(inputs: ep.Tensor) -> ep.Tensor:
                out = model(inputs)
                return out

            loss_fn = loss_fn_max
            gradient_step_sign = 1.0
        else:
            raise ValueError("unsupported criterion")

        if self.abs_stepsize is None:
            stepsize = self.rel_stepsize * epsilon
        else:
            stepsize = self.abs_stepsize

        if self.random_start:
            x = self.get_random_start(x0, epsilon)
            x = ep.clip(x, *model.bounds)
        else:
            x = x0

        for i_s in range(self.steps):
            val, gradients = self.value_and_grad(loss_fn, x)
            gradients = self.normalize(gradients, x=x, bounds=model.bounds)
            x = x + gradient_step_sign * stepsize * gradients
            x = self.project(x, x0, epsilon)
            x = ep.clip(x, *model.bounds)
        return restore_type(x)


def get_representations(model, data_loader, device, layer, n_inputs_max=np.inf, module=None, reload=True):
    print("Calculate representations...")
    inputs = []
    reps = []
    targets = []
    n_inputs = 0
    for step, (img, target) in enumerate(data_loader):
        # print("batch number: ", step, " of ", len(data_loader))
        residual_co = []
        model_input = img.to(device)
        inputs.append(model_input.clone())
        for l in range(layer):
            block = model.blocks[l]

            model_input = block(model_input)

        reps.append(model_input.detach())

        targets.append(target)
        n_inputs = n_inputs + target.shape[0]
        if n_inputs >= n_inputs_max:
            break

    inputs = torch.cat(inputs).cpu()
    reps = torch.cat(reps).cpu()
    targets = torch.cat(targets).cpu()

    return inputs, reps, targets


def rep_PCA(X, k):
    pca = PCA(n_components=k)
    pca.fit(X)
    print('PCA done and explained %s' % np.sum(pca.explained_variance_ratio_))
    Y = pca.transform(X)
    return Y


def rep_2d(inputs, reps, targets, n_points=None, apply_pca=True, avg_pixels=False, method='tsne'):
    n_samples = targets.shape[0]
    if n_points == None:
        n_points = n_samples

    d_inputs = inputs.reshape(n_samples, -1)

    if avg_pixels:
        reps = torch.mean(reps, (2, 3))  # spatial mean pooling

    d_reps = reps.reshape(n_samples, -1)

    if apply_pca:
        if d_reps.shape[1] > 100:
            d_reps = rep_PCA(d_reps, 100)

    print("Doing %s..." % method)
    if method == 't-sne':
        tsne_reps = manifold.TSNE(perplexity=50)

        # t_inputs = tsne_inputs.fit_transform(d_inputs[:n_points,:])
        t_reps = tsne_reps.fit_transform(d_reps[:n_points, :])
    else:
        reducer = umap.UMAP(random_state=42)
        reducer.fit(d_reps[:n_points, :])
        t_reps = reducer.transform(d_reps[:n_points, :])

    # tSNE_plot(opt, t_inputs, targets[:n_points], class_names, fig_name_ext = 'input')
    return t_reps, targets[:n_points]


def plot_2d(t_data, targets, class_names, plot_labels=True, title=None, xlim=None, ylim=None, marker_size=50,
            no_border=False, STORE_GRAPH=None):
    plt.close('all')
    plt.rcParams.update({'font.size': 25,
                         'lines.linewidth': 2,
                         'lines.linestyle': '-',
                         'lines.markersize': 8})

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    # color_targets = [colors[t] for t in targets]
    cmap2 = ListedColormap(colors)
    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(20, 15))

    scatter = plt.scatter(t_data[:, 0], t_data[:, 1], s=marker_size, c=targets, cmap=cmap2)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    if plot_labels:
        plt.legend(handles=scatter.legend_elements()[0],
                   labels=class_names.values(),
                   prop={'size': 18})
    if title is not None:
        if no_border:
            fig.patch.set_visible(False)
            ax.axis('off')
        else:
            plt.title(title)

        try:
            fig.savefig(op.join(STORE_GRAPH, title + '.png'))
            fig.savefig(op.join(STORE_GRAPH, title + '.png'), format='svg')
        except:
            print('---Save not available')


def unravel_index(indices: torch.LongTensor, shape) -> torch.LongTensor:
    """Converts flat indices into unraveled coordinates in a target shape.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).
    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


def max_activation(reps, n_neurons=4, n_max=10):
    s = reps.shape
    neuron_inds = np.random.choice([i for i in range(s[1])], size=(n_neurons), replace=False)
    responses = reps[:, neuron_inds, :, :].permute(1, 0, 2, 3).reshape(n_neurons, -1)  # n_neurons, b*x*y

    patches_list = []
    for n in range(n_neurons):
        inds_flattened = responses[n].argsort(descending=True)  # b*x*y (sorted by value, largest first)
        inds = unravel_index(inds_flattened, (s[0], s[2], s[3]))  # b*x*y, 3 (time, x, y)
        patches = inds[:n_max].clone()  # n_max, 3 (max. activating patches)

        ctr = 0
        tns = []
        tns.append(patches[0][0])
        for npatch in range(1, n_max):
            tnp = patches[npatch][0]
            while tnp in tns:
                ctr += 1
                patches[npatch][:] = inds[n_max + ctr]
                tnp = patches[npatch][0]
            tns.append(tnp)

        patches_list.append(patches)
    return patches_list


def imgs_patches(model, layer, indexes):
    sizes = [1]
    operations = []
    size = 1
    for l in range(layer - 1, -1, -1):
        conv = model.blocks[l].layer

        d_conv = conv.dilation[0]
        s_conv = conv.stride[0]
        k_conv = conv.kernel_size[0]
        p_conv = conv.padding[0]

        pool = model.blocks[l].pool
        k_pool = pool.kernel_size
        s_pool = pool.stride

        size = (size - 1) * s_pool + k_pool

        size = (size - 1) * s_conv + (k_conv - 1) * d_conv + 1

        sizes.append(size)
        operations.append(lambda x1, x2, s=s_pool, k=k_pool: (x1 * s - s // 2, (x2 - 1) * s + s // 2))
        operations.append(lambda x1, x2, p=p_conv, d=d_conv, s=s_conv, k=k_conv: (
        x1 * d * s - k // 2, (x2 - 1) * d * s + (k - k // 2)))

    patches = []
    for n in range(len(indexes)):
        patches_n = []
        for b, w, h in indexes[n]:
            h1 = int(h)
            h2 = int(h1 + 1)
            w1 = int(w)
            w2 = int(w1 + 1)
            for op in operations:
                h1, h2 = op(h1, h2)
                w1, w2 = op(w1, w2)
            patches_n.append([b, (w1, w2), (h1, h2)])
        patches.append(patches_n)

    return patches


def plot_patches(imgs, patches, layer, time=0, STORE_GRAPH=None):
    n_examples = len(patches)
    n_plots = len(patches[0])
    fig, axes = plt.subplots(nrows=n_examples, ncols=n_plots, figsize=(20, int(20 * n_examples / n_plots)), sharex=True,
                             sharey=True)  # , gridspec_kw={'wspace': 0.05})

    for i in range(n_examples):
        for j in range(n_plots):
            # b, (w0, w1), (h0, h1) = patches[i][j]
            b, (h0, h1), (w0, w1) = patches[i][j]
            ax = axes[i][j]

            ax.imshow(imgs[b].permute(1, 2, 0))

            rect = Rectangle((w0, h0), (w1 - w0), (h1 - h0), linewidth=1, edgecolor='r', facecolor='none')

            # Add the patch to the Axes
            ax.add_patch(rect)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
    plt.show()
    try:
        fig.savefig(op.join(STORE_GRAPH, 'patch_activation_' + str(layer) + '_' + str(time) + '.png', format='png'))
        fig.savefig(op.join(STORE_GRAPH, 'patch_activation_' + str(layer) + '_' + str(time) + '.svg', format='svg'))
    except:
        print('---Save not available')


def get_param(dict_):
    def p(d):
        parms = {}
        for key, value in d.items():
            if isinstance(value, dict):
                for x, y in p(value).items():
                    parms.update({key + '/' + x: y})
            else:
                parms.update({key: value})
        return parms

    return p(dict_)


def load_results(exp_path, folder):
    try:
        result = pd.read_csv(op.join(exp_path, folder, 'progress.csv'))

        with open(op.join(exp_path, folder, 'params.json')) as f:
            data = json.load(f)

        for param, value in get_param(data).items():
            result[param] = value
    except:
        print(folder)
        result = []
    return result


# def get_data(exp):
#     exp_path = op.join(SEARCH, exp)
#     folders = [f for f in os.listdir(exp_path) if op.isdir(op.join(exp_path, f))]
#     data = [load_results(exp_path, f) for f in folders]
#     data = [x for x in data if isinstance(x, pd.DataFrame)]
#     return data

def get_data(exp, return_folders=False):
    exp_path = op.join(SEARCH, exp)
    folders = [f for f in os.listdir(exp_path) if op.isdir(op.join(exp_path, f))]
    data = [load_results(exp_path, f) for f in folders]
    df_data = []
    used_folders = []
    for idx_x, x in enumerate(data):
        if isinstance(x, pd.DataFrame):
            df_data.append(x)
            used_folders.append(folders[idx_x])
    assert len(df_data) == len(used_folders)
    if return_folders:
        return df_data, used_folders
    else:
        return df_data


# def error(data, bars=16, wts='test_acc'):
#     if 'convergence' in data[0].columns:
#         for d in data:
#             d.rename(columns={'convergence': 'R1'}, inplace=True)
#     conv_acc = pd.DataFrame([d.iloc[-1][['R1', wts]] for d in data])
#     nb_r1 = conv_acc['R1'].max()
#     off = nb_r1 / bars / 2
#     conv_acc['bars'] = conv_acc.apply(lambda x: min(nb_r1, max(0, int((x['R1'] + off) / nb_r1 * bars) * nb_r1 / bars)),
#                                       1)
#     conv_acc = conv_acc.sort_values('bars')
#     error_acc = conv_acc[['bars', wts]].groupby('bars').agg(['mean', 'std']).fillna(method='ffill')
#
#     return error_acc


def error(data, bars=16, wts='test_acc'):
    if 'convergence' in data[0].columns:
        convergence_metric = 'convergence'
    else:
        convergence_metric = 'R1'
    conv_acc = pd.DataFrame([d.iloc[-1][[convergence_metric, wts]] for d in data])
    nb_r1 = conv_acc[convergence_metric].max()
    off = nb_r1/bars/2
    conv_acc['bars'] = conv_acc.apply(lambda x: min(nb_r1, max(0,int((x[convergence_metric]+off)/nb_r1*bars) * nb_r1 / bars)), 1)
    conv_acc = conv_acc.sort_values('bars')
    error_acc = conv_acc[['bars', wts]].groupby('bars').agg(['mean','std']).fillna(method='ffill')
    return error_acc


# def extract_data(data, features=None, wts='test_acc'):
#     if features is None:
#         features = ['b0/layer/t_invert', 'R1', wts]
#     conv_acc = pd.DataFrame([d.iloc[-1][features] for d in data])
#     conv_acc = conv_acc.groupby(features[:-2]).agg({wts: ['mean', 'std']}).droplevel(0, 1).reset_index()
#     conv_acc['t'] = 1 / conv_acc['b0/layer/t_invert']
#     conv_acc = conv_acc.sort_values('b0/layer/t_invert')
#     return conv_acc


def extract_data(data, features=None, wts='test_acc'):
    if features is None:
        features = ['b0/layer/softness', 'b0/layer/t_invert', 'R1', wts, 'dataset_unsup/seed']
    conv_acc = pd.DataFrame([d.iloc[-1][features] for d in data])
    if 'b0/layer/t_invert' in features:
        conv_acc = conv_acc.groupby(features[:-3]).agg({i: ['mean', 'std'] for i in features[-3:-1]})
        conv_acc.columns = conv_acc.columns.map('_'.join)
        conv_acc = conv_acc.reset_index()
        conv_acc['t'] = 1 / conv_acc['b0/layer/t_invert']
        conv_acc = conv_acc.sort_values('t')
    return conv_acc

# def extract_data(data, features=None, wts='test_acc'):
#     if features is None:
#         features = [wts]
#     conv_acc = pd.DataFrame([d.iloc[-1][features] for d in data])
#     return conv_acc


def load_data(exps, t='t1'):
    datas = []
    for exp in exps:
        data=[]
        _, log = load_logs(exp)
        for b in log.sup[t].batch[:-1]:
            d = b.get_numpy()
            d[:, 2] *= 100/ d[:, 0]
            data.extend(d)
        datas.append(np.array(data))
    return datas

def moving_avg(x, rolling_window=None):
    if rolling_window is None:
        rolling_window = max(50, int(len(x)/30))
    cumsum, moving_aves = [0], []

    for i, y in enumerate(x, 1):
        cumsum.append(cumsum[i-1] + y)
        if i > rolling_window:
            r = min(len(cumsum), rolling_window)
            moving_ave = (cumsum[i] - cumsum[i-r])/r
            moving_aves.append(moving_ave)
    return np.array(rolling_window * [moving_aves[0]] + moving_aves)


def plot_filter(filters, rows=10, cols=8, title=None):
    if filters.shape[0] < rows * cols:
        if filters.shape[0] <= cols:
            rows = 2
            cols = int(filters.shape[0] / 2)
        elif filters.shape[0] <= rows:
            cols = int(filters.shape[0] / rows)
        else:
            rows = int(filters.shape[0] / cols)

    if cols > 2:

        fig, axes = plt.subplots(
            rows,
            cols,
            figsize=(20, int(20 * rows / cols))
        )

        vmin = filters[:(rows + 1) * cols].min()
        vmax = filters[:(rows + 1) * cols].max()

        for row in range(rows):
            for col in range(cols):
                i = row * cols + col
                filter = filters[i]

                if len(filter.shape) == 3:
                    filter -= np.min(filter)  # Normalize
                    filter /= np.max(filter)
                    axes[row, col].imshow(filter, cmap='coolwarm')  # , vmin=vmin, vmax=vmax)
                else:
                    axes[row, col].imshow(filter, cmap='coolwarm', vmin=vmin, vmax=vmax)
                axes[row, col].axis("off")
                # axes[row, col].title.set_text('{i},r: {r:.2f}'.format(i=i, r=np.sum(np.abs(filter) ** 2)))
    else:
        fig, axes = plt.subplots(1, filters.shape[0], figsize=(20, 6))
        for row in range(filters.shape[0]):
            filter = filters[row]

            axes[row].imshow(filter, cmap='coolwarm', vmin=0, vmax=1)
            axes[row].axis("off")

    if title is not None:
        plt.title(title)

    plt.show()

    try:
        fig.savefig(op.join(STORE_GRAPH, title + '.png'))
        fig.savefig(op.join(STORE_GRAPH, title + '.png'), format='svg')
    except:
        print('---Save not available')