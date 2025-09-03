# %%
import random
import torch
from src import NodeRegression, NodeClassification, GraphClassification, GraphRegression
from dataset import (
    QM9DataModule_mu,
    QM9DataModule_Alpha,
    QM9DataModule_HOMO,
    QM9DataModule_LUMO,
    QM9DataModule_Gap,
    QM9DataModule_R2,
    QM9DataModule_ZPVE,
    QM9DataModule_U0,
    QM9DataModule_U,
    QM9DataModule_H,
    QM9DataModule_G,
    QM9DataModule_Cv,
)

# Define available tasks and models
tasks = {
    "node_regression": [],
    "node_classification": [],
    "graph_classification": [],
    "graph_regression": [
        QM9DataModule_Gap,
        QM9DataModule_mu,
        QM9DataModule_Alpha,
        QM9DataModule_HOMO,
        QM9DataModule_LUMO,
        QM9DataModule_R2,
        QM9DataModule_ZPVE,
        QM9DataModule_U0,
        QM9DataModule_U,
        QM9DataModule_H,
        QM9DataModule_G,
        QM9DataModule_Cv,
    ],
}

models = {
    "node_regression": NodeRegression,
    "node_classification": NodeClassification,
    "graph_classification": GraphClassification,
    "graph_regression": GraphRegression,
}


def generate_hyperparams(args):
    """Generates hyperparameters based on input arguments."""
    hyperparams = {
        "in_dims": [args.hidden_dims] * args.layers,
        "out_dims": [args.hidden_dims] * args.layers,
        "heads": [args.heads] * args.layers,
        "shifts": [
            random.uniform(args.shifts_min, args.shifts_max) for _ in range(args.layers)
        ],
        "widths": [
            random.uniform(args.shifts_min, args.shifts_max) for _ in range(args.layers)
        ],
    }
    return hyperparams


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()
