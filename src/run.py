import os
import json
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

import torch
import lightning.pytorch as pl
from data import Packets

from typing import Any

import criterions
import model as models
from config import get_config
from core import ArtifactsPlotter


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    pl.seed_everything(seed)


def get_criterion(criterion_config):
    return criterions.loss_factory.create(criterion_config["name"], **criterion_config)


def get_model_core(model_config):
    model = models.model_factory.create(model_config["name"], **model_config)

    if hasattr(model, "encoder") and isinstance(
        model.encoder[-2], torch.nn.modules.linear.LazyLinear
    ):
        model.encoder[-2].initialize_parameters(input=torch.rand(1, 5632))
        _ = model(torch.rand(1, 1, 100, 200), torch.rand(1, 128))
    return model


def get_plotter(config: dict[str, Any]):
    return ArtifactsPlotter(**config)


def get_device_config(device_n: int) -> dict[str, str]:
    device_config = {"accelerator": "gpu" if torch.cuda.is_available() else "cpu"}
    if device_config["accelerator"] == "gpu":
        device_config["devices"] = [device_n]
    return device_config


def get_model(config: dict[str, Any], criterion, plotter, checkpoint_path: str = None):
    if not checkpoint_path:
        model = models.PARCNetWrapper(
            **config["model"], plotter=plotter, criterion=criterion
        )
    else:
        model = models.PARCNetWrapper.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            **config["model"],
            plotter=plotter,
            criterion=criterion,
            config=config,
        )
    return model


def main(
    config: dict[str, Any],
    checkpoint_path: str = None,
    save_model: bool = False,
    device: int = 0,
    train: bool = False,
    evaluate: bool = False,
):
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    cello = Packets(**config["data"])
    cello.prepare_data()

    criterion = get_criterion(config["criterion"])
    plotter = get_plotter(
        {
            **config["plotter"],
            "dataset_waves_path": config["data"]["dataset_waves_path"],
            "valid_file_path": cello.valid_files[0].split("/")[-1],
        }
    )
    device_config = get_device_config(device)

    model = get_model(config, criterion, plotter, checkpoint_path)

    if train:
        logger = pl.loggers.TensorBoardLogger(
            config["experiment"]["logs_path"], config["experiment"]["name"]
        )
        trainer = pl.Trainer(
            max_epochs=config["experiment"]["epochs"], logger=logger, **device_config
        )

        trainer.fit(model, cello, ckpt_path=checkpoint_path)
        trainer.test(model, cello)
    if evaluate:
        logger = pl.loggers.TensorBoardLogger(
            config["experiment"]["logs_path"], config["experiment"]["name"]
        )
        trainer = pl.Trainer(
            max_epochs=config["experiment"]["epochs"], logger=logger, **device_config
        )

        trainer.test(model, cello, ckpt_path=checkpoint_path)
    if save_model:
        model = model.cpu()
        model.eval()

        save_path = (
            f"{config['experiment']['name']}-{checkpoint_path.split('/')[-3]}.pth"
        )
        model_to_serialize = (
            model.generator if "parcnet" in config["model"]["name"] else model.model
        )
        print(f"Saving model to {save_path}...")
        torch.jit.script(model_to_serialize).save(save_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config-file", type=str, required=True)
    parser.add_argument("-ckpt", "--checkpoint_path", type=str, required=False)
    parser.add_argument("-s", "--save-model", action="store_true", default=False)
    parser.add_argument("-t", "--train", action="store_true", default=False)
    parser.add_argument("-e", "--evaluate", action="store_true", default=False)
    parser.add_argument("-d", "--device", type=int, default=0)
    args = parser.parse_args()
    os.environ["CONFIG_FILE"] = args.config_file

    if not args.checkpoint_path and args.save_model:
        raise ValueError(
            "It is required to pass a '-ckpt' value when '-s' flag is passed."
        )

    config = get_config()
    seed_everything(config["experiment"]["seed"])
    main(
        config,
        args.checkpoint_path,
        args.save_model,
        args.device,
        args.train,
        args.evaluate,
    )