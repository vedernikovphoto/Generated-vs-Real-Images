from typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.augmentations import get_transforms, TransformFlags
from src.config import Config
from src.dataset import MaritimeDataset, DatasetConfig
from src.dataset_splitter import create_and_split_datasets
from utils.file_utils import read_df


class MaritimeDM(LightningDataModule):
    """
    Initialize the MaritimeDM data module.

    Args:
        config (Config): Configuration object containing parameters for the data module.
    """
    def __init__(self, config: Config):
        super().__init__()
        self._config = config
        self._augmentation_params = config.augmentation_params
        self.label_encoder = self._config.label_encoder
        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Prepare data by splitting and saving train, validation, and test datasets.
        """
        create_and_split_datasets(
            config=self._config,
            train_fraction=self._config.data_config.train_size,
            random_state=self._config.seed,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Set up datasets for training, validation, and testing.

        Args:
            stage (Optional[str]): Stage of the setup ('fit', 'test', etc.).
        """
        if stage == 'fit' or stage is None:
            df_train = read_df(self._config.data_config.data_path, 'train')
            df_valid = read_df(self._config.data_config.data_path, 'valid')

            train_transforms = get_transforms(
                aug_config=self._augmentation_params,
                width=self._config.data_config.width,
                height=self._config.data_config.height,
                flags=TransformFlags(augmentations=True),
            )

            val_transforms = get_transforms(
                aug_config=self._augmentation_params,
                width=self._config.data_config.width,
                height=self._config.data_config.height,
                flags=TransformFlags(augmentations=False),
            )

            train_config = DatasetConfig(transforms=train_transforms)
            val_config = DatasetConfig(transforms=val_transforms)
            self.train_dataset = MaritimeDataset(df_train, train_config)
            self.valid_dataset = MaritimeDataset(df_valid, val_config)

        if stage == 'test' or stage is None:
            df_test = read_df(self._config.data_config.data_path, 'test')
            test_transforms = get_transforms(
                aug_config=self._augmentation_params,
                width=self._config.data_config.width,
                height=self._config.data_config.height,
                flags=TransformFlags(augmentations=False),
            )
            test_config = DatasetConfig(transforms=test_transforms)
            self.test_dataset = MaritimeDataset(df_test, test_config)

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader configured for training with batch size,
                        workers, and shuffle options.
        """
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader configured for validation with batch size
                        and workers options.
        """
        return DataLoader(
            dataset=self.valid_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader for the test dataset.

        Returns:
            DataLoader: DataLoader configured for testing with batch size
                        and workers options.
        """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self._config.data_config.batch_size,
            num_workers=self._config.data_config.n_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            persistent_workers=True,
        )
