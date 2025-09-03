# train.py

import os
import argparse
from lightning import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    BatchSizeFinder,
    LearningRateFinder,
    RichProgressBar,
    # SpikeDetection,
)
from src.utils import tasks, models, generate_hyperparams


def main():
    parser = argparse.ArgumentParser(description="Train a model.")

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(models.keys()),
        help="Model type.",
    )
    parser.add_argument(
        "--task", type=str, required=True, choices=list(tasks.keys()), help="Task type."
    )
    parser.add_argument("--layers", type=int, default=3, help="Number of layers.")
    parser.add_argument(
        "--hidden_dims", type=int, default=64, help="Dimension of hidden layers."
    )
    parser.add_argument(
        "--heads", type=int, default=4, help="Number of heads in multi-head attention."
    )
    parser.add_argument(
        "--shifts_min", type=float, default=0.5, help="Minimum shift value."
    )
    parser.add_argument(
        "--shifts_max", type=float, default=3, help="Maximum shift value."
    )
    parser.add_argument(
        "--dataset_index",
        type=int,
        default=0,
        help="Index of the dataset in the task list.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1000,
        help="Maximum number of training epochs.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Accumulate gradients over N batches.",
    )

    args = parser.parse_args()

    seed_everything(42, workers=True)

    # Generate hyperparameters
    hyperparams = generate_hyperparams(args)

    # Initialize data module
    datamodule_list = tasks[args.task]
    if not datamodule_list:
        raise ValueError(f"No data modules available for task {args.task}")
    if args.dataset_index < 0 or args.dataset_index >= len(datamodule_list):
        raise ValueError(
            f"dataset_index {args.dataset_index} is out of bounds for task {args.task}"
        )
    datamodule_class = datamodule_list[args.dataset_index]
    datamodule = datamodule_class()
    hyperparams["x_dims"] = datamodule.x_dims
    hyperparams["y_dims"] = datamodule.y_dims

    # Initialize model
    model_class = models[args.model]
    model = model_class(hyperparams)

    # Configure logger
    run_name = (
        f"{args.model}_{args.task}_layers{args.layers}_hiddendim{args.hidden_dims}"
    )
    logger = WandbLogger(
        project="iclr2025",
        name=run_name,
        tags=[args.model, args.task],
        notes="Automated run with command-line arguments",
        config={
            "model": args.model,
            "task": args.task,
            "layers": args.layers,
            "hidden_dims": args.hidden_dims,
            "heads": args.heads,
            "shifts_min": args.shifts_min,
            "shifts_max": args.shifts_max,
            "dataset_index": args.dataset_index,
            "hyperparams": hyperparams,
        },
    )

    # Initialize trainer
    trainer = Trainer(
        default_root_dir="logs",
        max_epochs=args.max_epochs,
        logger=logger,
        callbacks=[
            BatchSizeFinder(),
            EarlyStopping(
                monitor="val_mae_epoch",
                patience=5,
                check_on_train_epoch_end=False,
            ),
            LearningRateFinder(),
            ModelCheckpoint(
                monitor="val_mae_epoch",
                dirpath=os.path.join("chkpts", run_name),
                filename="{epoch}-{val_mae_epoch:.4f}",
                save_top_k=3,
                mode="min",
            ),
            # RichProgressBar(),
        ],
        log_every_n_steps=1,
        deterministic=True,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Start training and testing
    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
