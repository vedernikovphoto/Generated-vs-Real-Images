import os
import pandas as pd
from typing import List


def get_image_files(folder_path: str, group_name: str) -> List[dict]:
    """
    Extracts image files from a folder and creates records with paths, labels, and groups.

    Args:
        folder_path (str): Path to the folder containing images.
        group_name (str): Group label for the images.

    Returns:
        List[dict]: List of dictionaries with image metadata.
    """
    return [
        {'image_path': os.path.join(folder_path, filename), 'label': 0, 'group': group_name}
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(('.jpg', '.png'))
    ]


def read_df(data_path: str, mode: str) -> pd.DataFrame:
    """
    Reads a DataFrame from a CSV file.

    Args:
        data_path (str): Path to the data directory.
        mode (str): Mode of the data ('train', 'valid', 'test').

    Returns:
        pd.DataFrame: DataFrame read from the specified CSV file.
    """
    return pd.read_csv(os.path.join(data_path, f'df_{mode}.csv'))
