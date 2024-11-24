import os
from typing import List, Tuple
import pandas as pd
from utils.dataframe_utils import create_real_dataframe
from utils.logging_utils import log_dataset_split_info
from sklearn.model_selection import GroupShuffleSplit


GROUP = 'group'


def stratify_group_split_subsets_with_folders(
    df: pd.DataFrame,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Splits the dataset into train, validation, and test subsets based on group labels.

    Args:
        df (pd.DataFrame): DataFrame with columns ['image_path', 'label', 'group'].
        train_fraction (float): Training fraction; the rest is split equally between validation and testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
            - Train, validation, and test DataFrames.
            - A dictionary with folder usage information for each split.
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=train_fraction, random_state=random_state)
    train_indices, temp_indices = next(gss.split(df, groups=df[GROUP]))
    train_df = df.iloc[train_indices]
    temp_df = df.iloc[temp_indices]

    gss_temp = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=random_state)
    valid_indices, test_indices = next(gss_temp.split(temp_df, groups=temp_df[GROUP]))
    valid_df = temp_df.iloc[valid_indices]
    test_df = temp_df.iloc[test_indices]

    folder_split_info = {
        'train_folders': train_df[GROUP].unique(),
        'valid_folders': valid_df[GROUP].unique(),
        'test_folders': test_df[GROUP].unique(),
    }

    train_df = train_df.drop(columns=[GROUP])
    valid_df = valid_df.drop(columns=[GROUP])
    test_df = test_df.drop(columns=[GROUP])

    return (
        train_df.reset_index(drop=True),
        valid_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
        folder_split_info,
    )


def split_and_save_real_datasets(
    real_dirs: List[str],
    data_path: str,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits real datasets into train, validation, and test subsets, saving them as CSV files.

    Args:
        real_dirs (List[str]): List of directories containing real image data.
        data_path (str): Path to save the resulting CSV files.
        train_fraction (float): Fraction of data for training.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Combined train, validation, and test DataFrames.
    """
    all_train = []
    all_valid = []
    all_test = []
    folder_split_summary = {}

    for idx, real_dir in enumerate(real_dirs):
        df_real = create_real_dataframe([real_dir])
        train_df, valid_df, test_df, folder_split_info = stratify_group_split_subsets_with_folders(
            df_real, train_fraction=train_fraction, random_state=random_state,
        )
        all_train.append(train_df)
        all_valid.append(valid_df)
        all_test.append(test_df)

        # Save separate CSV files for this Real dataset
        train_csv = os.path.join(data_path, f'real_{idx + 1}_train.csv')
        valid_csv = os.path.join(data_path, f'real_{idx + 1}_valid.csv')
        test_csv = os.path.join(data_path, f'real_{idx + 1}_test.csv')
        train_df.to_csv(train_csv, index=False)
        valid_df.to_csv(valid_csv, index=False)
        test_df.to_csv(test_csv, index=False)

        folder_split_summary[f'Real Dataset {idx + 1}'] = folder_split_info

        log_dataset_split_info(idx, train_df, valid_df, test_df, folder_split_info)

    return pd.concat(all_train), pd.concat(all_valid), pd.concat(all_test)
