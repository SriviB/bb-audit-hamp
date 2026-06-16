"""
Utility functions to load and transform datasets
"""
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, MNIST, CIFAR100
from torch.utils.data import TensorDataset
import os
import numpy as np

def load_colored_mnist(root='./', split='train', seed=0, majority_pct=0.995):
    """
    Binary Colored MNIST (even digits vs odd digits).

    Class 0 = even digits {0,2,4,6,8}, class 1 = odd digits {1,3,5,7,9}.
    Uses all 60k training / 10k test samples.

    majority_pct fraction of class 0 is red; (1-majority_pct) is blue.
    Symmetrically, majority_pct of class 1 is blue; (1-majority_pct) is red.

    Subgroups:
      0 = class 0 (even), red   (majority for class 0)
      1 = class 0 (even), blue  (minority for class 0)
      2 = class 1 (odd),  red   (minority for class 1)
      3 = class 1 (odd),  blue  (majority for class 1)

    Returns X (N,3,28,28), y (N,), subgroups (N,), out_dim=2.
    """
    os.makedirs(f'{root}/data/colored_mnist', exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = MNIST(root=f'{root}/data/colored_mnist', train=(split == 'train'),
                    download=True, transform=transform)

    xs, ys_list = [], []
    for x, y in dataset:
        xs.append(x)
        ys_list.append(0 if y % 2 == 0 else 1)   # even→0, odd→1
    X_gray = torch.stack(xs)                              # (N, 1, 28, 28)
    y = torch.tensor(ys_list, dtype=torch.long)
    N = len(y)

    rng = np.random.default_rng(seed)
    color = np.zeros(N, dtype=np.int64)  # 0=red, 1=blue
    y_np = y.numpy()
    minority_pct = 1.0 - majority_pct
    for cls, p_red in [(0, majority_pct), (1, minority_pct)]:
        idx = np.where(y_np == cls)[0]
        n_red = max(1, round(p_red * len(idx)))
        cls_color = np.concatenate([
            np.zeros(n_red, dtype=np.int64),
            np.ones(len(idx) - n_red, dtype=np.int64),
        ])
        rng.shuffle(cls_color)
        color[idx] = cls_color

    # subgroup = class * 2 + color: 0=c0_red, 1=c0_blue, 2=c1_red, 3=c1_blue
    subgroups = torch.tensor(y_np * 2 + color, dtype=torch.long)

    # Colorize: broadcast gray to 3 channels, then zero non-dominant channels
    X_rgb = X_gray.expand(-1, 3, -1, -1).clone()   # (N, 3, 28, 28)
    is_red = torch.tensor(color == 0)
    is_blue = ~is_red
    X_rgb[is_red, 1] = 0.0    # red → zero G
    X_rgb[is_red, 2] = 0.0    # red → zero B
    X_rgb[is_blue, 0] = 0.0   # blue → zero R
    X_rgb[is_blue, 1] = 0.0   # blue → zero G

    return X_rgb, y, subgroups, 2


def load_data(data_name, n_df, root='./', split='train'):
    os.makedirs(f'{root}/data', exist_ok=True)
    DATA_ROOT = f'{root}/data/{data_name}'

    if data_name == 'mnist':
        # load MNIST dataset
        MNIST_MEAN = (0.1307,)
        MNIST_STD_DEV = (0.3081,)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD_DEV)
        ])

        dataset = MNIST(
            root=DATA_ROOT, train=split=='train', download=True, transform=transform)

        out_dim = 10
    elif data_name == 'cifar10':
        # load CIFAR-10 dataset
        CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
        CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
        ])

        dataset = CIFAR10(
            root=DATA_ROOT, train=split=='train', download=True, transform=transform)
        
        out_dim = 10
    elif data_name == 'cifar100':
        # load CIFAR-100 dataset
        CIFAR100_MEAN = (0.5074, 0.4867, 0.4411)
        CIFAR100_STD_DEV = (0.2011, 0.1987, 0.2025)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD_DEV),
        ])

        dataset = CIFAR100(
            root=DATA_ROOT, train=split=='train', download=True, transform=transform)
        
        out_dim = 100
    elif data_name == 'purchase':
        # load purchase dataset
        npz_path = f'{DATA_ROOT}/purchase100.npz'

        data = np.load(npz_path)
        X, y = data['features'], data['labels']

        if len(y.shape) > 1 and y.shape[1] > 1:
            y = np.argmax(y, axis=1)

        X, y = torch.from_numpy(X), torch.from_numpy(y)

        n = len(y)
        split_idx = int(0.8 * n)
        if split == 'train':
            X, y = X[:split_idx], y[:split_idx]
        else:
            X, y = X[split_idx:], y[split_idx:]

        X = X.float()

        dataset = TensorDataset(X, y)
        out_dim = len(torch.unique(y))
    elif data_name == 'tiny_shakespeare':
        # load tiny shakespeare dataset
        input_file_path = f'{DATA_ROOT}/input.txt'
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()

        chars = sorted(list(set(text)))
        str_to_int = {ch: i for i, ch in enumerate(chars)}
        vocab_size = len(chars)

        data_ids = torch.tensor([str_to_int[ch] for ch in text], dtype=torch.long)

        split_idx = int(0.8 * len(data_ids))
        if split == 'train':
            data_ids = data_ids[:split_idx]
        else:
            data_ids = data_ids[split_idx:]

        X = data_ids[:-1]
        y = data_ids[1:]

        dataset = TensorDataset(X, y)
        out_dim = vocab_size
    elif data_name == 'colored_mnist':
        X, y, _, out_dim = load_colored_mnist(root=root, split=split)
        return X, y, out_dim
    elif os.path.exists(f'{DATA_ROOT}'):
        # load pre-processed local data
        X, y = np.load(f'{DATA_ROOT}/X_{split}.npy'), np.load(f'{DATA_ROOT}/y_{split}.npy')
        X, y = torch.from_numpy(X), torch.from_numpy(y)

        dataset = TensorDataset(X, y)
        out_dim = len(y.unique())
    else:
        raise Exception(f'Dataset {data_name} not configured and {DATA_ROOT} not found')
    
    # load neighboring dataset D-
    n_df = len(dataset) if n_df is None or n_df < 0 else n_df # load full dataset if n_df is None
    shuffle = n_df != len(dataset) # only shuffle dataset if full dataset is not loaded
    tmp_loader = torch.utils.data.DataLoader(dataset, batch_size=n_df, shuffle=shuffle)

    X, y = next(iter(tmp_loader))
    X, y = X, y

    return X, y, out_dim