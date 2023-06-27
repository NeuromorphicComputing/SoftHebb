import torch
import torch.nn as nn
from typing import Generator, Union

try:
    from utils import init_weight, normalize, activation, unsup_lr_scheduler
except:
    from hebb.utils import init_weight, normalize, activation, unsup_lr_scheduler
import einops
from tabulate import tabulate


class HebbHardLinear(nn.Module):
    def __init__(
            self,
            in_features: int,
            n_neurons: int,
            lebesgue_p: int,
            weight_distribution: str,
            weight_range: float,
            weight_offset: float,
            lr_scheduler: str,
            bias: bool = False
    ) -> None:
        """
        Hard Winner take all implementation
        """

        super().__init__()

        self.stat = torch.zeros(3, n_neurons)

        self.learning_update = False
        self.was_update = True

        self.in_features = in_features
        self.n_neurons = n_neurons
        self.lebesgue_p = lebesgue_p

        self.register_buffer(
            'weight',
            init_weight((n_neurons, in_features), weight_distribution, weight_range, weight_offset)
        )

        self.register_buffer("delta_w", torch.zeros_like(self.weight), persistent=False)

        self.register_buffer(
            "rad",
            torch.ones(n_neurons),
            persistent=False
        )
        self.get_radius()

        self.lr_scheduler_config = lr_scheduler.copy()
        self.lr_adaptive = self.lr_scheduler_config['adaptive']

        self.reset()

        self.conv = 0

        self.register_buffer('bias', None)

    def stat_wta(self):
        count = self.stat.clone()
        count[2:] = (100 * self.stat[2:].t() / self.stat[2:].sum(1)).t()
        count = count[:, :20]
        x = list(range(count.shape[1]))
        y = [['{lr:.1e}'.format(lr=lr) for lr in count[0].tolist()]] + [['{x:.2f}'.format(x=x) for x in y.tolist()] for
                                                                        y in count[1:]]
        return tabulate(y, headers=x, tablefmt='orgtbl')

    def reset(self):
        if self.lr_adaptive:
            self.register_buffer("lr", torch.ones_like(self.weight), persistent=False)
            self.lr_scheduler = unsup_lr_scheduler(lr=self.lr_scheduler_config['lr'],
                                                   nb_epochs=self.lr_scheduler_config['nb_epochs'],
                                                   ratio=self.lr_scheduler_config['ratio'],
                                                   speed=self.lr_scheduler_config['speed'],
                                                   div=self.lr_scheduler_config['div'],
                                                   decay=self.lr_scheduler_config['decay'])

            self.update_lr()
        else:
            self.lr_scheduler = unsup_lr_scheduler(lr=self.lr_scheduler_config['lr'],
                                                   nb_epochs=self.lr_scheduler_config['nb_epochs'],
                                                   ratio=self.lr_scheduler_config['ratio'],
                                                   speed=self.lr_scheduler_config['speed'],
                                                   div=self.lr_scheduler_config['div'],
                                                   decay=self.lr_scheduler_config['decay'])
            self.lr = next(self.lr_scheduler)

    def get_pre_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the preactivation or the current of the hebbian layer
        ----------
        x : torch.Tensor
            Input
        pre_x : torch.Tensor
            Pre activation
        Returns
        -------
        pre_x : torch.Tensor
            Pre_activation of the hebbian layer
        """
        pre_x = torch.matmul(x,
                             (torch.sign(self.weight) * torch.abs(self.weight) ** (self.lebesgue_p - 1)).t()
                             )

        if self.bias is not None:
            pre_x = torch.add(pre_x, self.bias)

        return pre_x

    def get_lr(self):
        if self.lr_adaptive:
            return self.lr.mean().cpu()
        return self.lr

    def get_wta(self, pre_x: torch.Tensor) -> torch.Tensor:
        """
        Compute the hard winner take all
        ----------
        pre_x : torch.Tensor
            Input
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer
        """
        wta = nn.functional.one_hot(pre_x.argmax(dim=1), num_classes=pre_x.shape[1]).to(
            torch.float)
        self.stat[2] += wta.sum(0).cpu()
        return wta

    def forward(self, x: torch.Tensor, return_x_wta: bool = False) -> torch.Tensor:
        """
        Compute output of the layer (forward pass).
        Parameters
        ----------
        x : torch.Tensor
            Input. Expected to be of shape (batch_size, ...), where ... denotes an arbitrary
            sequence of dimensions, with product equal to in_features.
        """
        if False:
            x = 10 * nn.functional.normalize(x)

        pre_x = self.get_pre_activations(x)

        # If propagation of preAcitvations only no need to do the rest
        if not self.learning_update and not return_x_wta:
            return pre_x

        wta = self.get_wta(pre_x)

        if return_x_wta:
            return pre_x, wta

        if self.learning_update:
            self.plasticity(x, pre_x, wta)

        return pre_x

    def train(self, mode: bool = True) -> None:
        """
        Set the learning update to the mode expected.
        mode:True --> training

        mode:False --> predict
        """
        self.learning_update = mode

    def delta_weight(
            self,
            x: torch.Tensor,
            pre_x: torch.Tensor,
            wta: torch.Tensor, ) -> torch.Tensor:
        """
        Compute the change of weights
        Parameters
        ----------
        x : torch.Tensor
            x. Input (batch_size, in_features).
        pre_x : torch.Tensor
            pre_x. Linear transformation of the input (batch_size, in_features).
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).
        Returns
        -------
            delta_weight : torch.Tensor
        """
        # ---Compute change of weights---#

        yx = torch.matmul(wta.t(), x)  # Overlap between winner take all and inputs
        yu = torch.multiply(wta, pre_x)
        yu = torch.sum(yu.t(), dim=1).unsqueeze(1)
        # Overlap between preactivation and winner take all
        # Results are summed over batches, resulting in a shape of (output_size,)
        delta_weight = yx - yu.view(-1, 1) * self.weight

        # ---Normalize---#
        nc = torch.abs(delta_weight).amax()  # .amax(1, keepdim=True)
        delta_weight.div_(nc + 1e-30)

        return delta_weight

    def plasticity(
            self,
            x: torch.Tensor,
            pre_x: torch.Tensor = None,
            wta: torch.Tensor = None) -> None:
        """
        Update weight and bias accordingly to the plasticity computation
        Parameters
        ----------
        x : torch.Tensor
            x. Input (batch_size, in_features).
        pre_x : torch.Tensor
            pre_x. Conv2d transformation of the input (batch_size, in_features).
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).

        """
        if pre_x is None:
            pre_x = self._conv_forward(x, self.weight, self.bias)
            # for some algo (krotov) pre_x and Conv2d trans are different
            pre_x = self.get_pre_x(x, pre_x)
            wta = self.get_wta(pre_x)

        self.delta_w = self.delta_weight(x, pre_x, wta)
        # self.weight.add_(self.lr * delta_weight)

        # self.update()

        if self.bias is not None:
            self.delta_b = self.delta_bias(wta)

    def update(self):
        """
        Update weight and bias accordingly to the plasticity computation
        Returns
        -------

        """
        self.weight.add_(self.lr * self.delta_w)
        self.was_update = True

        if self.bias is not None:
            self.bias.add_(self.lr * self.lrb * self.delta_b)
            # self.bias.clip_(-1, 0)
        self.update_lr()

    def update_lr(self) -> None:
        if self.lr_adaptive:
            norm = self.get_radius()

            nc = 1e-10

            # lr_amplitude = next(self.lr_scheduler)

            lr_amplitude = self.lr_scheduler_config['lr']

            lr = lr_amplitude * torch.pow(torch.abs(norm - torch.ones_like(norm)) + nc,
                                          self.lr_scheduler_config['power_lr'])

            # lr = lr.clip(max=lr_amplitude)

            self.stat[0] = lr.clone()

            self.lr = einops.repeat(lr, 'o -> o i', i=self.in_features)
        else:
            self.lr = next(self.lr_scheduler)
            self.stat[0] = self.lr

    def get_radius(self):
        if self.was_update:
            weight = self.weight.view(self.weight.shape[0], -1)
            self.rad = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
            self.was_update = False
        return self.rad

    def radius(self) -> float:
        """
        Returns
        -------
        radius : float
        """

        meanb = torch.mean(self.bias) if self.bias is not None else 0.
        stdb = torch.std(self.bias) if self.bias is not None else 0.
        weight = self.weight.view(self.weight.shape[0], -1)
        mean = torch.mean(weight, axis=1)
        mean_weight = torch.mean(mean)
        std_weigh = torch.std(weight)
        norm = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
        self.stat[1] = norm
        mean_radius = torch.mean(norm)
        std_radius = torch.std(norm)
        max_radius = torch.amax(torch.abs(norm - mean_radius * torch.ones_like(norm)))
        mean2_radius = torch.mean(torch.abs(norm - mean_radius * torch.ones_like(norm)))

        return 'MB:{mb:.3e}/SB:{sb:.3e}/MW:{m:.3e}/SW:{s:.3e}/MR:{mean:.3e}/SR:{std:.3e}/MeD:{mean2:.3e}/MaD:{max:.3e}'.format(
            mb=meanb,
            sb=stdb,
            m=mean_weight,
            s=std_weigh,
            mean=mean_radius,
            std=std_radius,
            mean2=mean2_radius,
            max=max_radius) + '\n' + self.stat_wta()

    def convergence(self) -> float:
        """
        Returns
        -------
        convergence : float
            Metric of convergence as the nb of filter closed to 1
        """
        weight = self.weight.view(self.weight.shape[0], -1)

        norm = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
        # mean_radius = torch.mean(norm)
        conv = torch.mean(torch.abs(norm - torch.ones_like(norm)))

        R1 = torch.sum(torch.abs(norm - torch.ones_like(norm)) < 5e-3)
        return float(conv.cpu()), int(R1.cpu())

    def extra_repr2(self) -> str:
        return 'in_features={}, out_features={}, lebesgue_p={},  bias={}'.format(
            self.in_features, self.n_neurons, self.lebesgue_p, self.bias is not None
        )

    def extra_repr(self) -> str:
        return self.extra_repr2()

    def __label__(self):
        s = '{in_features}{n_neurons}{lebesgue_p}'
        return 'H' + s.format(**self.__dict__)


class HebbHardKrotovLinear(HebbHardLinear):
    def __init__(
            self,
            in_features: int,
            n_neurons: int,
            lebesgue_p: int,
            weight_distribution: str,
            weight_range: float,
            weight_offset: float,
            lr_scheduler: Generator,
            bias: bool = False,
            delta: float = 0.05,
            ranking_param: int = 2
    ) -> None:
        """
        Krotov implementation from the HardLinear class
        """

        super(HebbHardKrotovLinear, self).__init__(in_features, n_neurons, lebesgue_p, weight_distribution,
                                                   weight_range, weight_offset, lr_scheduler, bias)

        self.delta = delta
        self.ranking_param = ranking_param
        self.stat = torch.zeros(4, n_neurons)

    def extra_repr(self):
        s = ', ranking_param=%s, delta=%s' % (self.ranking_param, self.delta)
        return self.extra_repr2() + s

    def get_wta(self, pre_x: torch.Tensor) -> torch.Tensor:
        """
        Compute the krotov winner take all
        ----------
        pre_x : torch.Tensor
            pre_x
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer
        """
        _, ranks = pre_x.sort(descending=True, dim=1)
        wta = nn.functional.one_hot(pre_x.argmax(dim=1), num_classes=pre_x.shape[1]).to(
            torch.float)

        self.stat[2] += wta.sum(0).cpu()
        # wta = wta - self.delta * nn.functional.one_hot(ranks[:, self.ranking_param-1], num_classes=pre_x.shape[1])
        batch_indices = torch.arange(pre_x.size(0))
        _, ranking_indices = pre_x.topk(self.ranking_param, dim=1)
        wta[batch_indices, ranking_indices[batch_indices, self.ranking_param - 1]] = -self.delta

        self.stat[3] += torch.histc(torch.tensor(ranking_indices[batch_indices, self.ranking_param - 1]),
                                    bins=self.out_channels, min=0,
                                    max=self.out_channels - 1).cpu()
        # print(wta[batch_indices, ranking_indices[batch_indices, 0]].mean())
        return wta


class HebbSoftLinear(HebbHardLinear):
    def __init__(
            self,
            in_features: int,
            n_neurons: int,
            lebesgue_p: int,
            weight_distribution: str,
            weight_range: float,
            weight_offset: float,
            lr_scheduler: Generator,
            lr_bias: float,
            bias: bool = False,
            activation_fn: str = 'exp',
            t_invert: float = 12
    ) -> None:
        """
        Soft implementation from the HardLinear class
        """
        super(HebbSoftLinear, self).__init__(in_features, n_neurons, lebesgue_p, weight_distribution,
                                             weight_range, weight_offset, lr_scheduler, bias)

        self.activation_fn = activation_fn
        self.t_invert = torch.tensor(t_invert)

        if bias:
            self.register_buffer('bias', torch.ones(n_neurons) \
                                 * torch.log(torch.tensor(1 / n_neurons)) / self.t_invert
                                 )  # uniform initial priors, and acount for softmax's T_invert

        self.lrb = torch.tensor(1 / t_invert)

    def extra_repr(self):
        s = ', t_invert=%s, bias=%s, lr_bias=%s' % (
        float(self.t_invert), not self.bias is None, round(float(self.lrb), 4))
        return self.extra_repr2() + s

    def get_wta(self, pre_x: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft winner take all
        ----------
        pre_x : torch.Tensor
            pre_x
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer
        """
        wta = activation(pre_x, t_invert=self.t_invert, activation_fn=self.activation_fn, normalize=True)
        self.stat[2] += wta.sum(0).cpu()
        return wta

    def delta_bias(self, wta: torch.Tensor) -> None:
        """
        Compute the change of Bias
        Parameters
        ----------
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).
        """
        batch_size = wta.shape[0]
        if self.activation_fn == 'exp':
            ebb = torch.exp(self.t_invert * self.bias)  # e^(bias*t_invert)
            # ---Compute change of bias---#
            delta_bias = (torch.sum(wta, dim=0) - ebb * batch_size) / ebb
        elif self.activation_fn == 'relu':
            delta_bias = (torch.sum(wta, dim=0) - wta.shape[0] * self.bias - batch_size)  # eta *    (y-w-1)

        nc = torch.abs(delta_bias).amax()  # .amax(1, keepdim=True)
        delta_bias.div_(nc + 1e-30)

        return delta_bias


class HebbSoftKrotovLinear(HebbSoftLinear):
    def __init__(
            self,
            in_features: int,
            n_neurons: int,
            lebesgue_p: int,
            weight_distribution: str,
            weight_range: float,
            weight_offset: float,
            lr_scheduler: Generator,
            lr_bias: float,
            bias: bool = False,
            delta: float = 0.05,
            ranking_param: int = 2,
            activation_fn: str = 'exp',
            t_invert: float = 12
    ) -> None:
        """
        Krotov implementation from the SoftLinear class
        """

        super(HebbSoftKrotovLinear, self).__init__(in_features, n_neurons, lebesgue_p, weight_distribution,
                                                   weight_range,
                                                   weight_offset, lr_scheduler, lr_bias, bias, activation_fn, t_invert)

        self.delta = delta
        self.ranking_param = ranking_param

        self.m_winner = []
        self.m_anti_winner = []
        self.mode = 0
        self.stat = torch.zeros(4, n_neurons)

    def extra_repr(self):
        s = ', t_invert=%s, bias=%s, lr_bias=%s' % (
        float(self.t_invert), not self.bias is None, round(float(self.lrb), 4))
        s += ', ranking_param=%s, delta=%s' % (self.ranking_param, self.delta)
        return self.extra_repr2() + s

    def get_wta(self, pre_x: torch.Tensor) -> torch.Tensor:
        """
        Compute the soft krotov winner take all
        ----------
        pre_x : torch.Tensor
            pre_x
        Returns
        -------
        wta : torch.Tensor
            preactivation or the current of the hebbian layer
        """
        batch_size, out_channels = pre_x.shape
        # pre_x = pre_x - torch.mean(pre_x, axis=1, keepdims=True)
        # pre_x[pre_x < 0] = -float("Inf")
        wta = activation(pre_x, t_invert=self.t_invert, activation_fn=self.activation_fn, normalize=True)
        self.stat[2] += wta.sum(0).cpu()
        # print(wta.sum(0).cpu())
        batch_indices = torch.arange(pre_x.size(0))
        if self.mode == 0:
            wta = -wta
            # _, ranking_indices = pre_x_flat.topk(1, dim=1)
            # ranking_indices = ranking_indices[batch_indices,0]
            ranking_indices = torch.argmax(pre_x, dim=1)
            wta[batch_indices, ranking_indices] *= -1
            self.m_winner.append(wta[batch_indices, ranking_indices].mean().cpu())
            self.m_anti_winner.append(1 - self.m_winner[-1])
        if self.mode == 1:
            _, ranking_indices = pre_x.topk(self.ranking_param, dim=1)

            self.m_anti_winner.append(
                wta[batch_indices, ranking_indices[batch_indices, self.ranking_param - 1]].mean().cpu())
            self.m_winner.append(wta[batch_indices, ranking_indices[batch_indices, 0]].mean().cpu())

            # print(wta[batch_indices, ranking_indices[batch_indices, self.ranking_param-1]].mean())
            wta[batch_indices, ranking_indices[batch_indices, self.ranking_param - 1]] *= -self.delta
            # print(wta[batch_indices, ranking_indices[batch_indices, self.ranking_param-1]].mean())
        # print(wta[batch_indices, ranking_indices[batch_indices, 0]].mean())
        if self.mode == 2:
            _, ranking_indices = pre_x.topk(self.ranking_param, dim=1)

            self.m_anti_winner.append(
                wta[batch_indices, ranking_indices[batch_indices, self.ranking_param - 1]].mean().cpu())
            self.m_winner.append(wta[batch_indices, ranking_indices[batch_indices, 0]].mean().cpu())

            # print(wta[batch_indices, ranking_indices[batch_indices, self.ranking_param-1]].mean())
            wta[batch_indices, ranking_indices[batch_indices, self.ranking_param - 1]] = -self.delta
        return wta

    def radius(self) -> float:
        """
        Returns
        -------
        radius : float
        """
        meanb = torch.mean(self.bias) if self.bias is not None else 0.
        stdb = torch.std(self.bias) if self.bias is not None else 0.
        weight = self.weight.view(self.weight.shape[0], -1)
        mean = torch.mean(weight, axis=1)
        mean_weight = torch.mean(mean)
        std_weigh = torch.std(weight)
        norm = torch.linalg.norm(weight, dim=1, ord=self.lebesgue_p)
        self.stat[1] = norm
        mean_radius = torch.mean(norm)
        std_radius = torch.std(norm)
        max_radius = torch.amax(torch.abs(norm - mean_radius * torch.ones_like(norm)))
        mean2_radius = torch.mean(torch.abs(norm - mean_radius * torch.ones_like(norm)))

        m_winner = torch.mean(torch.tensor(self.m_winner))
        m_anti_winner = torch.mean(torch.tensor(self.m_anti_winner))

        self.m_winner = []
        self.m_anti_winner = []

        return 'MB:{mb:.3e}/SB:{sb:.3e}/MW:{m:.3e}/SW:{s:.3e}/MR:{mean:.3e}/SR:{std:.3e}/MeD:{mean2:.3e}/MaD:{max:.3e}/MW:{m_winner:.3f}/MAW:{m_anti_winner:.3f}'.format(
            mb=meanb,
            sb=stdb,
            m=mean_weight,
            s=std_weigh,
            mean=mean_radius,
            std=std_radius,
            mean2=mean2_radius,
            max=max_radius,
            m_winner=m_winner,
            m_anti_winner=m_anti_winner
        ) + '\n' + self.stat_wta() + '\n'


class SupervisedSoftHebbLinear(HebbSoftKrotovLinear):
    def __init__(self, **kwargs):
        self.forward_store = {}
        self.async_updates = True  # TODO make this a parameter from preset
        super().__init__(**kwargs)

    def get_wta(self, pre_x: torch.Tensor) -> torch.Tensor:
        """ should not be called"""
        raise NotImplementedError

    def forward(
            self, x: torch.Tensor, return_x_wta: bool = False,
    ) -> torch.Tensor:
        """
        Compute output of the layer (forward pass).
        Parameters
        ----------
        x : torch.Tensor
            Input. Expected to be of shape (batch_size, ...), where ... denotes an arbitrary
            sequence of dimensions, with product equal to in_features.
        """

        pre_x = self.get_pre_activations(x)

        # If propagation of preAcitvations only no need to do the rest
        if not self.learning_update and not return_x_wta:
            return pre_x

        # if clamped_wta is None and not self.async_updates:
        #     wta = self.get_wta(pre_x) # we don't need to do this
        # else:
        #     wta = clamped_wta
        if not self.async_updates:
            wta = self.get_wta(pre_x)
            if return_x_wta:
                return pre_x, wta

        if self.learning_update:
            # this does happen, we should change it, or change the behaviour based on this
            # pdb.set_trace()  # we shouldn't perform the learning update here, but later when we have the targets
            if self.async_updates:
                self.forward_store['x'] = x
                self.forward_store['pre_x'] = pre_x
            else:
                self.plasticity(x, pre_x, wta)
        return pre_x

    def plasticity(
            self,
            x: torch.Tensor,
            pre_x: torch.Tensor = None,
            wta: torch.Tensor = None) -> None:
        """
        Update weight and bias accordingly to the plasticity computation
        Parameters
        ----------
        x : torch.Tensor
            x. Input (batch_size, in_features).
        pre_x : torch.Tensor
            pre_x. Conv2d transformation of the input (batch_size, in_features).
        wta : torch.Tensor
            wta. Winner take all (batch_size, in_features).

        """
        if pre_x is None:
            raise ValueError  # although actually we could recompute, but throw error for now

        self.delta_w = self.delta_weight(x, pre_x, wta)

        if self.bias is not None:
            self.delta_b = self.delta_bias(wta)

        # My idea is to call this at some point, where I have the labels too !


def select_linear_layer(
        params) -> Union[HebbHardLinear, HebbHardKrotovLinear, HebbSoftLinear, HebbSoftKrotovLinear,
                         SupervisedSoftHebbLinear]:
    """
    Select the appropriate from a set of params
    ----------
    params : torch.Tensor
        wta. Winner take all (batch_size, in_features).
    Returns
        -------
        layer : bio
            preactivation or the current of the hebbian layer

    """
    if params['softness'] == 'hard':
        layer = HebbHardLinear(
            in_features=params['in_channels'],
            n_neurons=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            weight_offset=params['weight_init_offset'],
            lr_scheduler=params['lr_scheduler'],
            bias=params['add_bias'])
    elif params['softness'] == 'hardkrotov':
        layer = HebbHardKrotovLinear(
            in_features=params['in_channels'],
            n_neurons=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            weight_offset=params['weight_init_offset'],
            lr_scheduler=params['lr_scheduler'],
            bias=params['add_bias'],
            delta=params['delta'],
            ranking_param=params['ranking_param'])
    elif params['softness'] == 'soft':
        layer = HebbSoftLinear(
            in_features=params['in_channels'],
            n_neurons=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            weight_offset=params['weight_init_offset'],
            lr_scheduler=params['lr_scheduler'],
            lr_bias=params['lr_bias'],
            bias=params['add_bias'],
            activation_fn=params['soft_activation_fn'],
            t_invert=params['t_invert'])
    elif params['softness'] == 'softkrotov':
        layer = HebbSoftKrotovLinear(
            in_features=params['in_channels'],
            n_neurons=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            weight_offset=params['weight_init_offset'],
            lr_scheduler=params['lr_scheduler'],
            lr_bias=params['lr_bias'],
            bias=params['add_bias'],
            delta=params['delta'],
            ranking_param=params['ranking_param'],
            activation_fn=params['soft_activation_fn'],
            t_invert=params['t_invert'])
    elif params['softness'] == 'supervisedsoftkrotov':
        layer = SupervisedSoftHebbLinear(
            in_features=params['in_channels'],
            n_neurons=params['out_channels'],
            lebesgue_p=params['lebesgue_p'],
            weight_distribution=params['weight_init'],
            weight_range=params['weight_init_range'],
            weight_offset=params['weight_init_offset'],
            lr_scheduler=params['lr_scheduler'],
            lr_bias=params['lr_bias'],
            bias=params['add_bias'],
            delta=params['delta'],
            ranking_param=params['ranking_param'],
            activation_fn=params['soft_activation_fn'],
            t_invert=params['t_invert'])
    else:
        raise ValueError
    return layer
