import pandas as pd
import os
from typing import List
from utils.file_utils import get_image_files
from src.config import SplittingConfig


def create_real_dataframe(real_dirs: List[str]) -> pd.DataFrame:
    """
    Creates a DataFrame containing image paths, labels, and group labels for real images.

    Args:
        real_dirs (List[str]): List of directories containing real image data.

    Returns:
        pd.DataFrame: A DataFrame with columns ['image_path', 'label', 'group'],
                      where 'label' is 0 for real images.
    """
    image_records = []
    for dir_index, real_dir in enumerate(real_dirs):
        for folder_name in os.listdir(real_dir):
            folder_path = os.path.join(real_dir, folder_name)
            if os.path.isdir(folder_path):
                group_name = f'{dir_index}_{folder_name}'
                image_records.extend(get_image_files(folder_path, group_name))

    return pd.DataFrame(image_records)


def create_generated_dataframe(gen_split: SplittingConfig, generated_dir: str) -> pd.DataFrame:
    """
    Creates a DataFrame containing image paths, labels, and split labels for generated images.

    Args:
        gen_split (SplittingConfig): Configuration for generated dataset splits.
        generated_dir (str): Directory containing generated image data.

    Returns:
        pd.DataFrame: A DataFrame with columns ['image_path', 'label', 'split'],
                      where 'label' is 1 for generated images and 'split' is
                      assigned based on filename indices.
    """
    image_records = []
    all_files = os.listdir(generated_dir)
    filenames = [f for f in all_files if f.lower().endswith(('.jpg', '.png'))]
    filenames.sort()

    for filename in filenames:
        filepath = os.path.join(generated_dir, filename)
        index_str = filename.split('-')[0]
        index = int(index_str)
        if index <= gen_split.train_split_limit - 1:
            split = 'train'
        elif gen_split.train_split_limit <= index <= gen_split.val_split_limit - 1:
            split = 'valid'
        else:
            split = 'test'
        image_records.append({'image_path': filepath, 'label': 1, 'split': split})

    return pd.DataFrame(image_records)
