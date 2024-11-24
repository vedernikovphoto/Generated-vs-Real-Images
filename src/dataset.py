from typing import Optional, Tuple
import pandas as pd
import cv2
from dataclasses import dataclass
from torch.utils.data import Dataset
import albumentations as albu
import torch


@dataclass
class DatasetConfig:
    """
    Configuration class for dataset parameters.

    Attributes:
        transforms (Optional[albu.BaseCompose]): Optional albumentations transforms to be applied.
    """
    transforms: Optional[albu.BaseCompose] = None


class MaritimeDataset(Dataset):
    """
    Custom Dataset class for loading and transforming images and labels.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'image_path' and 'label'.
            - 'image_path': str, path to the image file.
            - 'label': int, corresponding label for the image.
        config (DatasetConfig): Configuration object specifying dataset parameters,
            including optional image transformations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: DatasetConfig,
    ):
        """
        Initialize the MaritimeDataset object.

        Args:
            df (pd.DataFrame): DataFrame with image paths and labels.
            config (DatasetConfig): Dataset configuration containing transformation settings.
        """
        self.df = df
        self.transforms = config.transforms

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve an image and its label from the dataset at the specified index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The processed image as a tensor.
                - The label as a long tensor.

        Raises:
            FileNotFoundError: If the image file at the specified path does not exist.
        """
        row = self.df.iloc[idx]
        image_path = row['image_path']
        label = int(row['label'])

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f'Image not found at path: {image_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        data_dict = {'image': image}
        if self.transforms:
            data_dict = self.transforms(**data_dict)

        return data_dict['image'], torch.tensor(label, dtype=torch.long)

    def __len__(self) -> int:
        """
        Get the number of items in the dataset.

        Returns:
            int: Total number of items in the dataset.
        """
        return len(self.df)
