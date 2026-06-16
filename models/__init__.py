from .lr import LR
from .cnn import CNN
from .wideresnet import WideResNet
from .lstm import LSTM
from .mlp import MLP

Models = {
    'lr': LR,
    'cnn': CNN,
    'wideresnet': WideResNet,
    'lstm': LSTM,
    'mlp': MLP
}