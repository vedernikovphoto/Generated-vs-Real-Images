import os
import argparse
import logging


def filter_png_files(files) -> list[str]:
    """
    Filters and returns only PNG files from a given list of file names.

    Args:
        files (list[str]): List of file names to filter.

    Returns:
        list[str]: List of file names that have the '.png' extension.
    """
    return [img for img in files if img.lower().endswith('.png')]


def keep_every_nth_image(top_folder, n) -> None:
    """
    Keeps every Nth image in subfolders of the specified folder and deletes the rest.

    Args:
        top_folder (str): Path to the folder containing subfolders of images.
        n (int): Interval for keeping images.
    """
    logging.info('Processing ABOships dataset...')

    subfolders = [
        os.path.join(top_folder, d)
        for d in os.listdir(top_folder)
        if os.path.isdir(os.path.join(top_folder, d))
    ]

    for subfolder in subfolders:
        all_files = os.listdir(subfolder)
        images = filter_png_files(all_files)

        images.sort()

        images_to_keep = set()
        for idx, image in enumerate(images):
            if idx % n == 0:
                images_to_keep.add(image)

        for img in images:
            if img not in images_to_keep:
                image_path = os.path.join(subfolder, img)
                os.remove(image_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{message}', style='{')
    parser = argparse.ArgumentParser(description='Retain every Nth image in subfolders of a given folder.')
    parser.add_argument('--top_folder', required=True, help='Path to the folder containing subfolders of images.')
    parser.add_argument('--frame_step', type=int, default=4, help='Step size for frame deletion.')
    args = parser.parse_args()

    keep_every_nth_image(args.top_folder, args.frame_step)
