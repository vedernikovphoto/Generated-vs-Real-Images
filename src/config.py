from typing import List, Dict
from omegaconf import OmegaConf
from pydantic import BaseModel


class LossConfig(BaseModel):
    """
    Configuration for a loss function.

    Attributes:
        name (str): The name of the loss function.
        weight (float): The weight of the loss function in the total loss.
        loss_fn (str): The name or path of the loss function.
        loss_kwargs (dict): Additional keyword arguments for the loss function.
    """
    name: str
    weight: float
    loss_fn: str
    loss_kwargs: dict


class RegularizationConfig(BaseModel):
    """
    Configuration for regularization techniques.

    Attributes:
        l1_lambda (float): Regularization strength for L1 regularization.
    """
    l1_lambda: float


class DataConfig(BaseModel):
    """
    Configuration for the dataset.

    Attributes:
        data_path (str): Path to the data directory.
        batch_size (int): Number of samples per batch.
        n_workers (int): Number of worker threads for data loading.
        train_size (float): Proportion of the data to be used for training.
        width (int): Width to resize the images to.
        height (int): Height to resize the images to.
    """
    data_path: str
    batch_size: int
    n_workers: int
    train_size: float
    width: int
    height: int


class SplittingConfig(BaseModel):
    """
    Configuration for the Dataset Split.

    Attributes:
        train_split_limit (int): The maximum index for the training split (inclusive).
        val_split_limit (int): The maximum index for the validation split (inclusive).
    """
    train_split_limit: int
    val_split_limit: int


class AugmentationConfig(BaseModel):
    """
    Configuration for data augmentation.

    Attributes:
        hue_shift_limit (int): Maximum change in hue during augmentation.
        sat_shift_limit (int): Maximum change in saturation during augmentation.
        val_shift_limit (int): Maximum change in value (brightness) during augmentation.
        brightness_limit (float): Range for random brightness adjustment.
        contrast_limit (float): Range for random contrast adjustment.
        shift_limit (float): Maximum shift fraction for affine transformations.
        scale_limit (float): Maximum scaling fraction for affine transformations.
        rotate_limit (float): Maximum rotation angle in degrees for affine transformations.
        blur_limit (List[int]): Minimum and maximum kernel size for Gaussian blur.
        elastic_alpha (float): Alpha value for elastic deformation.
        elastic_sigma (float): Sigma value for elastic deformation.
        grid_distort_num_steps (int): Number of distortion steps for grid distortion.
        grid_distort_limit (float): Maximum distortion magnitude for grid distortion.
        optical_distort_limit (float): Maximum distortion magnitude for optical distortion.
        optical_shift_limit (float): Maximum pixel shift for optical distortion.
        coarse_dropout_max_holes (int): Maximum number of holes for coarse dropout.
        coarse_dropout_max_height (int): Maximum height of holes for coarse dropout.
        coarse_dropout_max_width (int): Maximum width of holes for coarse dropout.
        coarse_dropout_min_holes (int): Minimum number of holes for coarse dropout.
        coarse_dropout_min_height (int): Minimum height of holes for coarse dropout.
        coarse_dropout_min_width (int): Minimum width of holes for coarse dropout.
        coarse_dropout_fill_value (int): Fill value for coarse dropout holes.
        gauss_noise_var_limit (List[float]): Range for Gaussian noise variance.
        motion_blur_limit (List[int]): Range for motion blur kernel size.
        random_gamma_limit (List[int]): Range for random gamma adjustment.
    """
    hue_shift_limit: int
    sat_shift_limit: int
    val_shift_limit: int
    brightness_limit: float
    contrast_limit: float
    shift_limit: float
    scale_limit: float
    rotate_limit: float
    blur_limit: List[int]
    elastic_alpha: float
    elastic_sigma: float
    grid_distort_num_steps: int
    grid_distort_limit: float
    optical_distort_limit: float
    optical_shift_limit: float
    coarse_dropout_max_holes: int
    coarse_dropout_max_height: int
    coarse_dropout_max_width: int
    coarse_dropout_min_holes: int
    coarse_dropout_min_height: int
    coarse_dropout_min_width: int
    coarse_dropout_fill_value: int
    gauss_noise_var_limit: List[float]
    motion_blur_limit: List[int]
    random_gamma_limit: List[int]


class Config(BaseModel):
    """
    Main configuration class for the project.

    Attributes:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        data_config (DataConfig): Dataset configuration parameters.
        augmentation_params (AugmentationConfig): Data augmentation parameters.
        regularization (RegularizationConfig): Regularization settings.
        gen_split (SplittingConfig): Configuration for generated dataset splits.
        n_epochs (int): Number of training epochs.
        num_classes (int): Number of output classes in the dataset.
        accelerator (str): Type of accelerator to use.
        device (int): Device ID for the accelerator.
        seed (int): Random seed for reproducibility.
        log_every_n_steps (int): Frequency of logging during training.
        patience (int): Epochs to wait for improvement before stopping.
        monitor_metric (str): Metric to monitor during training.
        monitor_mode (str): Optimization mode for the monitored metric.
        mdl_parameters (dict): Model-specific parameters.
        optimizer (str): Optimizer to use.
        optimizer_kwargs (dict): Additional optimizer parameters.
        scheduler (str): Scheduler to use for learning rate adjustment.
        scheduler_kwargs (dict): Additional scheduler parameters.
        losses (List[LossConfig]): List of loss functions with their configurations.
        label_encoder (Dict[str, int]): Mapping of class labels to integers.

    Methods:
        from_yaml(cls, path: str) -> 'Config': Load configuration from a YAML file.
    """
    project_name: str
    experiment_name: str
    data_config: DataConfig
    augmentation_params: AugmentationConfig
    regularization: RegularizationConfig
    gen_split: SplittingConfig
    n_epochs: int
    num_classes: int
    accelerator: str
    device: int
    seed: int
    log_every_n_steps: int
    patience: int
    monitor_metric: str
    monitor_mode: str
    mdl_parameters: dict
    optimizer: str
    optimizer_kwargs: dict
    scheduler: str
    scheduler_kwargs: dict
    losses: List[LossConfig]
    label_encoder: Dict[str, int]

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        """
        Load configuration from a YAML file.

        Args:
            path (str): Path to the YAML configuration file.

        Returns:
            Config: Loaded configuration object.
        """
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)
