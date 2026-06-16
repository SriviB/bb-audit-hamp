import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_shape=None, out_dim=10, hidden_dim=256):
        super(MLP, self).__init__()

        # stole from lr
        if in_shape is None:
            in_dim = 3072
        elif type(in_shape) == int:
            # D
            in_dim = in_shape
        elif len(in_shape) == 2:
            # B x D
            in_dim = in_shape[1]
        else:
            # B x C x H x W
            in_dim = in_shape[1] * in_shape[2] * in_shape[3]

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)

        self.dropout = nn.Dropout(0.2)
        self.embeddings = None

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        self.embeddings = x.clone().detach()

        out = self.fc3(x)
        return out