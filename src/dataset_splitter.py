import os
import pandas as pd
import logging
from utils.dataframe_utils import create_generated_dataframe
from utils.logging_utils import log_dataset_info
from utils.splits_utils import split_and_save_real_datasets
from src.config import Config


logging.basicConfig(level=logging.INFO, format='{message}', style='{')
SPLIT = 'split'
IMAGES = 'images'


def create_and_split_datasets(
    config: Config,
    train_fraction: float = 0.8,
    random_state: int = 42,
) -> None:
    """
    Creates, splits, and saves combined datasets for real and generated images.

    Args:
        config (Config): Configuration object containing parameters for the data module.
        train_fraction (float): Fraction of data for training.
        random_state (int): Random seed for reproducibility.
    """
    data_path = config.data_config.data_path
    real_dirs = [
        os.path.join(data_path, 'real', 'ABOshipsDataset', 'Seaships'),
        os.path.join(data_path, 'real2', IMAGES),
        os.path.join(data_path, 'real3', IMAGES),
        os.path.join(data_path, 'real4', IMAGES),
    ]
    generated_dir = os.path.join(data_path, 'generated', IMAGES)

    # Split and save Real datasets
    train_real_df, valid_real_df, test_real_df = split_and_save_real_datasets(
        real_dirs, data_path, train_fraction=train_fraction, random_state=random_state,
    )

    # Split and save Generated dataset
    df_generated = create_generated_dataframe(config.gen_split, generated_dir)
    train_generated_df = df_generated[df_generated[SPLIT] == 'train'].drop(columns=[SPLIT])
    valid_generated_df = df_generated[df_generated[SPLIT] == 'valid'].drop(columns=[SPLIT])
    test_generated_df = df_generated[df_generated[SPLIT] == 'test'].drop(columns=[SPLIT])

    train_generated_df.to_csv(os.path.join(data_path, 'gen_train.csv'), index=False)
    valid_generated_df.to_csv(os.path.join(data_path, 'gen_valid.csv'), index=False)
    test_generated_df.to_csv(os.path.join(data_path, 'gen_test.csv'), index=False)

    log_dataset_info('Generated', train_generated_df, valid_generated_df, test_generated_df)

    # Combine Real and Generated datasets
    train_df = (
        pd.concat([train_real_df, train_generated_df]).
        sample(frac=1, random_state=random_state).
        reset_index(drop=True)
    )
    valid_df = (
        pd.concat([valid_real_df, valid_generated_df]).
        sample(frac=1, random_state=random_state).
        reset_index(drop=True)
    )
    test_df = (
        pd.concat([test_real_df, test_generated_df]).
        sample(frac=1, random_state=random_state).
        reset_index(drop=True)
    )

    # Save combined datasets
    train_df.to_csv(os.path.join(data_path, 'df_train.csv'), index=False)
    valid_df.to_csv(os.path.join(data_path, 'df_valid.csv'), index=False)
    test_df.to_csv(os.path.join(data_path, 'df_test.csv'), index=False)

    log_dataset_info('Final combined', train_df, valid_df, test_df)
