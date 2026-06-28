from .lr import LR
from .cnn import CNN
from .wideresnet import WideResNet
from .wideresnet_nonprivate import WideResNetNonPrivate
from .lstm import LSTM
from .mlp import MLP

Models = {
    'lr': LR,
    'cnn': CNN,
    'wideresnet': WideResNet,
    'wideresnet_np': WideResNetNonPrivate,
    'lstm': LSTM,
    'mlp': MLP
}