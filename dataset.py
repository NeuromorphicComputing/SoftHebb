import copy
import random

try:
    from utils import seed_init_fn, DATASET
except:
    from hebb.utils import seed_init_fn, DATASET
import numpy as np
import os
import os.path as op
import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, STL10, ImageNet, ImageFolder
from typing import Optional, Any


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean


def imagenet_tf(width, height):
    return transforms.Compose([
        transforms.RandomSizedCrop((width, height)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def imagenet_test(width, height):
    return transforms.Compose([
        transforms.Resize((width, height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def advanced_transform(width, height):
    return transforms.Compose([
        transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=20 / 360)],
                               p=0.5),
        transforms.RandomApply([transforms.ColorJitter(saturation=1)], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(8),
        transforms.RandomApply(
            [transforms.Lambda(lambda x: TF.resize(x, (48 + random.randint(-6, 6), 48 + random.randint(-6, 6))))],
            p=0.3),
        transforms.RandomApply([transforms.RandomAffine(degrees=10, shear=10)], p=0.3),
        transforms.CenterCrop(40),
        transforms.RandomApply([transforms.RandomCrop((width, height))], p=0.5),
        transforms.CenterCrop((width, height)),
    ])


def crop_flip(width, height):
    return transforms.Compose(
        [
            transforms.RandomCrop(
                (width, height), padding=4, padding_mode="reflect"
            ),
            transforms.RandomHorizontalFlip(p=0.5),

        ]
    )


def select_dataset(dataset_config, device, dataset_path):
    test_transform = None
    val_indices = None
    split = dataset_config["split"] if "split" in dataset_config else "train"
    if dataset_config['name'] == 'CIFAR10':
        dataset_class = FastCIFAR10
        indices = list(range(50000))

        if dataset_config['augmentation']:
            dataset_train_class = AugFastCIFAR10
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = crop_flip(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = FastCIFAR10
            transform = None

    elif dataset_config['name'] == 'CIFAR100':
        dataset_class = FastCIFAR100
        indices = list(range(50000))

        if dataset_config['augmentation']:
            dataset_train_class = AugFastCIFAR100
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = crop_flip(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = FastCIFAR100
            transform = None

    elif dataset_config['name'] == 'MNIST':
        dataset_class = FastMNIST
        indices = list(range(60000))

        if dataset_config['augmentation']:
            dataset_train_class = AugFastMNIST
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = crop_flip(dataset_config['width'], dataset_config['height'])
            transform = AddGaussianNoise(std=dataset_config['noise_std'])
        else:
            dataset_train_class = FastMNIST
            transform = None

    elif dataset_config['name'] == 'FashionMNIST':
        dataset_class = FastFashionMNIST
        indices = list(range(60000))

        if dataset_config['augmentation']:
            dataset_train_class = AugFastFashionMNIST
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = crop_flip(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = FastFashionMNIST
            transform = None

    elif dataset_config['name'].startswith('ImageNette'):
        device = 'cpu'
        dataset_class = ImageNette
        indices = list(range(9469))

        if dataset_config['px'] == 'default':
            dataset_path = '/home/username/.fastai/data/imagenette2'
        elif dataset_config['px'] == 320:
            dataset_path = '/home/username/.fastai/data/imagenette2-320'
        else:
            dataset_path = os.path.join(DATASET, 'imagenette2-160')  # '/home/username/.fastai/data/imagenette2-160'

        if dataset_config['augmentation']:
            dataset_train_class = ImageNette
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = imagenet_tf(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = ImageNette
            transform = imagenet_test(dataset_config['width'], dataset_config['height'])
        test_transform = imagenet_test(dataset_config['width'], dataset_config['height'])

    elif dataset_config['name'].startswith('ImageNetV2'):
        device = 'cpu'
        dataset_class = ImageNetV2
        indices = list(range(10000))

        if dataset_config['name'][10:] == 'MatchedFrequency':
            dataset_path = '/scratch/hrodriguez/workspace/data/imagenetv2-matched-frequency-format-val'
        elif dataset_config['name'][10:] == 'Threshold07':
            dataset_path = '/scratch/hrodriguez/workspace/data/imagenetv2-threshold0.7-format-val'
        elif dataset_config['name'][10:] == 'TopImages':
            dataset_path = '/scratch/hrodriguez/workspace/data/imagenetv2-top-images-format-val'
        else:
            raise ValueError

        if dataset_config['augmentation']:
            dataset_train_class = ImageNetV2
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = imagenet_tf(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = ImageNetV2
            transform = imagenet_test(dataset_config['width'], dataset_config['height'])
        test_transform = imagenet_test(dataset_config['width'], dataset_config['height'])

    elif dataset_config['name'] == 'ImageNet':
        device = 'cpu'
        dataset_class = AugImageNet
        indices = list(range(1000000))

        dataset_path = '/scratch/datasets/ilsvrc12/'

        if dataset_config['augmentation']:
            dataset_train_class = AugImageNet
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = imagenet_tf(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = AugImageNet
            transform = imagenet_test(dataset_config['width'], dataset_config['height'])
        test_transform = imagenet_test(dataset_config['width'], dataset_config['height'])

    elif dataset_config['name'] == 'STL10':
        device = 'cpu'
        dataset_class = FastSTL10
        if split == 'train':
            indices = list(range(5000))
        elif split == 'unlabeled':
            indices = list(range(100000))
        else:
            indices = list(range(105000))

        if dataset_config['augmentation']:
            dataset_train_class = AugFastSTL10
            dataset_config['num_workers'] = 4
            device = 'cpu'
            transform = crop_flip(dataset_config['width'], dataset_config['height'])
        else:
            dataset_train_class = FastSTL10
            transform = None
    else:
        raise ValueError

    if not isinstance(dataset_config['training_class'], str):
        # we have to select indices up to the training_sample (trainign set size) otherwise the future origin_dataset
        # won't have enough indeces (it only stores the datapoints of the chosen training_class(es)
        # Another headache is that although you give training_sample, the validation set is taken from that
        # In the end: if want to validate, do it with only the same class(es)
        not_all_classes_samples = dataset_config['training_sample']
        if dataset_config['validation']:
            not_all_classes_samples += dataset_config['val_sample']
        not_all_classes_indices = indices[:not_all_classes_samples]
        indices = copy.deepcopy(not_all_classes_indices)

    if dataset_config['shuffle']:
        np.random.shuffle(indices)
        train_indices = indices[:dataset_config['training_sample']]

    if dataset_config['validation']:
        val_indices = indices[:dataset_config['val_sample']]
        train_indices = indices[
                        dataset_config['val_sample']:(dataset_config['training_sample'] + dataset_config['val_sample'])]
    return dataset_train_class, dataset_class, test_transform, transform, device, train_indices, val_indices, split, dataset_path


def make_data_loaders(dataset_config, batch_size, device, dataset_path=DATASET):
    """
     Load Mnist Dataset and create a dataloader

    Parameters
    ----------
    dataset_config : dict
        Configuration of the expected dataset
    batch_size: int
    dataset_path : str path
        Path to the dataset folder.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training dataloader.
    test_loader : torch.utils.data.DataLoader
        Testing dataloader.

    """
    g = torch.Generator()
    if dataset_config['seed'] is not None:
        seed_init_fn(dataset_config['seed'])
        g.manual_seed(dataset_config['seed'] % 2 ** 32)

    dataset_train_class, dataset_class, test_transform, transform, device, train_indices, val_indices, split, dataset_path = select_dataset(
        dataset_config, device, dataset_path)

    train_sampler = SubsetRandomSampler(train_indices, generator=g)
    origin_dataset = dataset_train_class(
        dataset_path,
        split=split,
        train=True,
        download=not dataset_config['name'] in ['ImageNet'],  # TODO: make this depend on whether dataset exists or not
        transform=transform,
        zca=dataset_config['zca_whitened'],
        device=device,
        train_class=dataset_config['training_class']
    )
    train_loader = torch.utils.data.DataLoader(dataset=origin_dataset,
                                               batch_size=batch_size,
                                               num_workers=dataset_config['num_workers'],
                                               sampler=train_sampler)

    if val_indices is not None:
        val_sampler = SubsetRandomSampler(val_indices)
        test_loader = torch.utils.data.DataLoader(dataset=origin_dataset,
                                                  batch_size=batch_size,
                                                  num_workers=dataset_config['num_workers'],
                                                  sampler=val_sampler)
    else:
        test_loader = torch.utils.data.DataLoader(
            dataset_class(
                dataset_path,
                split="val" if dataset_config['name'] in ['ImageNet', 'ImageNette',
                                                          'ImageNetV2MatchedFrequency'] else "test",
                train=False,
                zca=dataset_config['zca_whitened'],
                transform=test_transform,
                device=device
            ),
            batch_size=batch_size if dataset_config['name'] in ['STL10', 'ImageNet', 'ImageNette',
                                                                'ImageNetV2MatchedFrequency', 'ImageNetV2TopImages',
                                                                'ImageNetV2Threshold07'] else 1000,
            num_workers=dataset_config['num_workers'],
            shuffle=dataset_config['shuffle'],
        )
    return train_loader, test_loader


def whitening_zca(x: torch.Tensor, transpose=True, dataset: str = "CIFAR10"):
    path = op.join(DATASET, dataset + "_zca.pt")
    zca = None
    try:
        zca = torch.load(path, map_location='cpu')['zca']
    except:
        pass

    if zca is None:

        if transpose:
            x = x.copy().transpose(0, 3, 1, 2)

        x = x.copy().reshape(x.shape[0], -1)

        cov = np.cov(x, rowvar=False)

        u, s, v = np.linalg.svd(cov)

        SMOOTHING_CONST = 1e-1
        zca = np.dot(u, np.dot(np.diag(1.0 / np.sqrt(s + SMOOTHING_CONST)), u.T))
        zca = torch.from_numpy(zca).float()

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({'zca': zca}, path)

    return zca


# *************************************************** Imagenet-10 ***************************************************

class AugImageNet(ImageNet):
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        train = kwargs.pop('train', True)
        super().__init__(*args, **kwargs)


class ImageNette(ImageFolder):

    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        train = kwargs.pop('train', True)
        assert split in ['val', 'train']
        self.split = split

        super(ImageNette, self).__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self) -> str:
        return os.path.join(self.root, self.split)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


class ImageNetV2(ImageFolder):

    def __init__(self, root: str, split: str = 'train', download: Optional[str] = None, **kwargs: Any) -> None:
        root = self.root = os.path.expanduser(root)
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        train = kwargs.pop('train', True)
        assert split in ['test',
                         'val']  # although it's called val i think it' really a test, we don't use it for model dev
        self.split = split

        super(ImageNetV2, self).__init__(self.split_folder, **kwargs)

    @property
    def split_folder(self) -> str:
        return self.root

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


# *************************************************** STL-10 ***************************************************

class FastSTL10(STL10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        train = kwargs.pop('train', True)
        super().__init__(*args, **kwargs)

        mean = (0.4914, 0.48216, 0.44653)
        std = (0.247, 0.2434, 0.2616)

        # norm = transforms.Normalize(mean,  std)

        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)

        if train:
            if not isinstance(train_class, str):
                index_class = np.isin(self.labels, train_class)
                self.data = self.data[index_class]
                self.labels = np.array(self.labels)[index_class]
                self.len = self.data.shape[0]

        if zca:
            self.data = (self.data - mean) / std
            self.zca = whitening_zca(self.data, transpose=False, dataset=STL10)
            zca_whitening = transforms.LinearTransformation(self.zca, torch.zeros(self.zca.size(1)))
        self.data = torch.tensor(self.data, dtype=torch.float)

        # self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        # self.data = norm(self.data)
        if zca:
            self.data = zca_whitening(self.data)
            print("self.data.mean(), self.data.std()", self.data.mean(), self.data.std())

        # self.data = self.data.to(device)  # Rescale to [0, 1]

        # self.data = self.data.div_(CIFAR10_STD) #(NOT) Normalize to 0 centered with 1 std

        self.labels = torch.tensor(self.labels, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        return img, target


class AugFastSTL10(FastSTL10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """

        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# *************************************************** CIFAR-10 ***************************************************

class FastCIFAR10(CIFAR10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        mean = (0.4914, 0.48216, 0.44653)
        std = (0.247, 0.2434, 0.2616)

        norm = transforms.Normalize(mean, std)

        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)

        if self.train:
            if not isinstance(train_class, str):
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = np.array(self.targets)[index_class]
                self.len = self.data.shape[0]

        if zca:
            self.data = (self.data - mean) / std
            self.zca = whitening_zca(self.data)
            zca_whitening = transforms.LinearTransformation(self.zca, torch.zeros(self.zca.size(1)))
        self.data = torch.tensor(self.data, dtype=torch.float)

        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        # self.data = norm(self.data)
        if zca:
            self.data = zca_whitening(self.data)
            print("self.data.mean(), self.data.std()", self.data.mean(), self.data.std())

        # self.data = self.data.to(device)  # Rescale to [0, 1]

        # self.data = self.data.div_(CIFAR10_STD) #(NOT) Normalize to 0 centered with 1 std

        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.data[index]
        target = self.targets[index]

        return img, target


class AugFastCIFAR10(FastCIFAR10):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.transform(self.data[index])
        target = self.targets[index]

        return img, target


class FastCIFAR100(CIFAR100):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        mean = (0.4914, 0.48216, 0.44653)
        std = (0.247, 0.2434, 0.2616)

        norm = transforms.Normalize(mean, std)

        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255)

        if self.train:
            if not isinstance(train_class, str):
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = np.array(self.targets)[index_class]
                self.len = self.data.shape[0]
                print(self.len)

        if zca:
            self.data = (self.data - mean) / std
            self.zca = whitening_zca(self.data)
            zca_whitening = transforms.LinearTransformation(self.zca, torch.zeros(self.zca.size(1)))
        self.data = torch.tensor(self.data, dtype=torch.float)

        self.data = torch.movedim(self.data, -1, 1)  # -> set dim to: (batch, channels, height, width)
        # self.data = norm(self.data)
        if zca:
            self.data = zca_whitening(self.data)
            print(self.data.mean(), self.data.std())

        # self.data = self.data.to(device)  # Rescale to [0, 1]

        # self.data = self.data.div_(CIFAR10_STD) #(NOT) Normalize to 0 centered with 1 std

        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.data[index]
        target = self.targets[index]

        return img, target


class AugFastCIFAR100(FastCIFAR100):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.transform(self.data[index])
        target = self.targets[index]

        return img, target


# ***************************************************  MNIST ***************************************************

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)

        self.split = split

        if self.train:
            if not isinstance(train_class, str):
                print(train_class)
                self.targets = np.array(self.targets)
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = self.targets[index_class]
                self.len = self.data.shape[0]

        # Scale data to [0,1]
        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255).unsqueeze(1)

        self.targets = torch.tensor(self.targets, device=device)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class AugFastMNIST(FastMNIST):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.transform(self.data[index])
        target = self.targets[index]

        return img, target


# ***************************************************  FashionMNIST ***************************************************

class FastFashionMNIST(FashionMNIST):
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', "cpu")
        zca = kwargs.pop('zca', False)
        train_class = kwargs.pop('train_class', 'all')
        split = kwargs.pop('split', 'train')
        super().__init__(*args, **kwargs)
        self.split = split
        if self.train:
            if not isinstance(train_class, str):
                print(train_class)
                self.targets = np.array(self.targets)
                index_class = np.isin(self.targets, train_class)
                self.data = self.data[index_class]
                self.targets = self.targets[index_class]
                self.len = self.data.shape[0]

        # Scale data to [0,1]
        self.data = torch.tensor(self.data, dtype=torch.float, device=device).div_(255).unsqueeze(1)

        self.targets = self.targets.to(device)

        # Normalize it with the usual MNIST mean and std
        # self.data = self.data.sub_(0.1307).div_(0.3081)

        # Put both data and targets on GPU in advance

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        return img, target


class AugFastFashionMNIST(FastFashionMNIST):
    """
    Improves performance of training on CIFAR10 by removing the PIL interface and pre-loading on the GPU (2-3x speedup).

    Taken from https://github.com/y0ast/pytorch-snippets/tree/main/fast_mnist
    """

    def __getitem__(self, index: int):
        """
        Parameters
        ----------
        index : int
            Index of the element to be returned

        Returns
        -------
            tuple: (image, target) where target is the index of the target class
        """
        img = self.transform(self.data[index])
        target = self.targets[index]

        return img, target
