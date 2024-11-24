import albumentations as albu
from albumentations.pytorch import ToTensorV2
from src.config import AugmentationConfig
from dataclasses import dataclass


@dataclass
class TransformFlags:
    """
    Flags to enable or disable different types of transformations.

    Attributes:
        preprocessing (bool): If True, apply preprocessing transformations.
        augmentations (bool): If True, apply augmentation transformations.
        postprocessing (bool): If True, apply postprocessing transformations.
    """
    preprocessing: bool = True
    augmentations: bool = True
    postprocessing: bool = True


def get_transforms(
    aug_config: AugmentationConfig,
    width: int,
    height: int,
    flags: TransformFlags = None,
) -> albu.BaseCompose:
    """
    Get the data augmentation and preprocessing transformations.

    Args:
        aug_config (AugmentationConfig): Augmentation configuration object.
        width (int): Width to resize the image to.
        height (int): Height to resize the image to.
        flags (TransformFlags): Transform flags to toogle preprocessing, augmentations, and postprocessing.

    Returns:
        albu.BaseCompose: A composition of the specified transformations.
    """
    if flags is None:
        flags = TransformFlags()

    transforms = []

    if flags.preprocessing:
        transforms.append(albu.Resize(height=height, width=width))

    if flags.augmentations:
        transforms.extend(
            [
                albu.HorizontalFlip(p=0.5),
                albu.VerticalFlip(p=0.5),
                albu.HueSaturationValue(
                    hue_shift_limit=aug_config.hue_shift_limit,
                    sat_shift_limit=aug_config.sat_shift_limit,
                    val_shift_limit=aug_config.val_shift_limit,
                    p=0.5,
                ),
                albu.RandomBrightnessContrast(
                    brightness_limit=aug_config.brightness_limit,
                    contrast_limit=aug_config.contrast_limit,
                    p=0.5,
                ),
                albu.ShiftScaleRotate(
                    shift_limit=aug_config.shift_limit,
                    scale_limit=aug_config.scale_limit,
                    rotate_limit=aug_config.rotate_limit,
                    p=0.5,
                ),
                albu.GaussianBlur(
                    blur_limit=tuple(aug_config.blur_limit),
                    p=0.5,
                ),
                albu.RandomRotate90(p=0.5),
                albu.Transpose(p=0.5),
                albu.ElasticTransform(
                    alpha=aug_config.elastic_alpha,
                    sigma=aug_config.elastic_sigma,
                    p=0.5,
                ),
                albu.GridDistortion(
                    num_steps=aug_config.grid_distort_num_steps,
                    distort_limit=aug_config.grid_distort_limit,
                    p=0.5,
                ),
                albu.OpticalDistortion(
                    distort_limit=aug_config.optical_distort_limit,
                    shift_limit=aug_config.optical_shift_limit,
                    p=0.5,
                ),
                albu.CoarseDropout(
                    max_holes=aug_config.coarse_dropout_max_holes,
                    max_height=aug_config.coarse_dropout_max_height,
                    max_width=aug_config.coarse_dropout_max_width,
                    min_holes=aug_config.coarse_dropout_min_holes,
                    min_height=aug_config.coarse_dropout_min_height,
                    min_width=aug_config.coarse_dropout_min_width,
                    fill_value=aug_config.coarse_dropout_fill_value,
                    p=0.5,
                ),
                albu.GaussNoise(
                    var_limit=tuple(aug_config.gauss_noise_var_limit),
                    p=0.5,
                ),
                albu.MotionBlur(
                    blur_limit=tuple(aug_config.motion_blur_limit),
                    p=0.5,
                ),
                albu.RandomGamma(
                    gamma_limit=tuple(aug_config.random_gamma_limit),
                    p=0.5,
                ),
            ],
        )

    if flags.postprocessing:
        transforms.extend([albu.Normalize(), ToTensorV2()])

    return albu.Compose(transforms)