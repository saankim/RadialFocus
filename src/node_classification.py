import torch.nn as nn
from src.layers import MoireLayer, get_focus
import torchmetrics
import torch.nn.functional as F
from src.model import BaseModel


class NodeClassification(BaseModel):
    def __init__(self, hyperparams):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(hyperparams["x_dims"], 32),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, hyperparams["in_dims"][0]),
        )
        focus = get_focus("gaussian")
        self.moire_layers = nn.ModuleList(
            [
                MoireLayer(in_dims, out_dims, heads, focus, shift, width)
                for in_dims, out_dims, heads, shift, width in zip(
                    hyperparams["in_dims"],
                    hyperparams["out_dims"],
                    hyperparams["heads"],
                    hyperparams["shifts"],
                    hyperparams["widths"],
                )
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(hyperparams["out_dims"][-1], 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, hyperparams["y_dims"]),
        )
        self.loss_fn = nn.L1Loss()
        self.train_loss = torchmetrics.MeanAbsoluteError()
        self.val_loss = torchmetrics.MeanAbsoluteError()
        self.test_loss = torchmetrics.MeanAbsoluteError()
        self.lr = 1e-3

    def forward(self, x, adj, mask):
        x = self.input(x)
        for moire_layer in self.moire_layers:
            x = moire_layer(x, adj, mask)
        x = self.output(x)
        return F.sigmoid(x)
