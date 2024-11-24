import argparse
import os

import pytorch_lightning as pl
from clearml import Task
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.config import Config
from src.constants import EXPERIMENTS_PATH
from src.datamodule import MaritimeDM
from src.lightning_module import MaritimeModule


def arg_parse() -> argparse.Namespace:
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', type=str, help='config file')
    return parser.parse_args()


def train(config: Config) -> None:
    """
    Trains and evaluates the Maritime model.

    Sets up the data module, model, ClearML task, and PyTorch Lightning Trainer
    with necessary callbacks for training and testing.

    Args:
        config (Config): Training configuration object.

    Returns:
        None
    """
    datamodule = MaritimeDM(config)
    model = MaritimeModule(config)

    task = Task.init(
        project_name=config.project_name,
        task_name=f'{config.experiment_name}',
        auto_connect_frameworks=True,
    )
    task.connect(config.dict())

    experiment_save_path = os.path.join(EXPERIMENTS_PATH, config.experiment_name)
    os.makedirs(experiment_save_path, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=experiment_save_path,
        monitor=config.monitor_metric,
        mode=config.monitor_mode,
        save_top_k=1,
        filename=f'epoch_{{epoch:02d}}-{{{config.monitor_metric}:.3f}}',
    )
    trainer = pl.Trainer(
        max_epochs=config.n_epochs,
        accelerator=config.accelerator,
        devices=[config.device],
        log_every_n_steps=config.log_every_n_steps,
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor=config.monitor_metric, patience=config.patience, mode=config.monitor_mode),
            LearningRateMonitor(logging_interval='epoch'),
        ],
    )

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path='best', datamodule=datamodule)


if __name__ == '__main__':
    args = arg_parse()
    config = Config.from_yaml(args.config_file)
    pl.seed_everything(config.seed, workers=True)
    train(config)
