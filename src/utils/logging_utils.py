import logging
import pandas as pd


def log_dataset_split_info(
    idx: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    folder_split_info: dict,
) -> None:
    """
    Logs detailed information about the dataset splits and folder usage.

    Args:
        idx (int): Index of the dataset.
        train_df (pd.DataFrame): Training DataFrame.
        valid_df (pd.DataFrame): Validation DataFrame.
        test_df (pd.DataFrame): Test DataFrame.
        folder_split_info (dict): Dictionary containing folder usage information.
    """
    train_len = len(train_df)
    valid_len = len(valid_df)
    test_len = len(test_df)
    log_dataset_info_by_length(f'Dataset {idx + 1}', train_len, valid_len, test_len)

    train_folders_len = len(folder_split_info['train_folders'])
    valid_folders_len = len(folder_split_info['valid_folders'])
    test_folders_len = len(folder_split_info['test_folders'])

    logging.info(f'Folders used for Train: {train_folders_len}')
    logging.info(f'Folders used for Validation: {valid_folders_len}')
    logging.info(f'Folders used for Test: {test_folders_len}')

    train_folders = folder_split_info['train_folders']
    valid_folders = folder_split_info['valid_folders']
    test_folders = folder_split_info['test_folders']

    logging.info(f'Train Folders: {train_folders}')
    logging.info(f'Validation Folders: {valid_folders}')
    logging.info(f'Test Folders: {test_folders}')


def log_dataset_info(dataset_name: str, train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Logs the number of samples in the train, validation, and test datasets.

    Args:
        dataset_name (str): Name of the dataset being logged (e.g., "Generated", "Combined").
        train_df (pd.DataFrame): DataFrame for the training split.
        valid_df (pd.DataFrame): DataFrame for the validation split.
        test_df (pd.DataFrame): DataFrame for the test split.
    """
    logging.info(f'\n{dataset_name} dataset:')
    logging.info(f'Train samples: {len(train_df)}')
    logging.info(f'Valid samples: {len(valid_df)}')
    logging.info(f'Test samples: {len(test_df)}')


def log_dataset_info_by_length(dataset_name: str, train_len: int, valid_len: int, test_len: int) -> None:
    """
    Logs the number of samples in the train, validation, and test datasets.

    Args:
        dataset_name (str): Name of the dataset being logged (e.g., "Dataset 1").
        train_len (int): Number of samples in the training dataset.
        valid_len (int): Number of samples in the validation dataset.
        test_len (int): Number of samples in the test dataset.
    """
    logging.info(f'\n{dataset_name}:')
    logging.info(f'Train samples: {train_len}')
    logging.info(f'Valid samples: {valid_len}')
    logging.info(f'Test samples: {test_len}')
