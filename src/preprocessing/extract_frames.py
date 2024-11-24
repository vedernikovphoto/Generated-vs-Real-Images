import cv2
import os
import shutil
import argparse
import logging


def ensure_folder_exists(folder: str) -> None:
    """
    Ensures that the specified folder is empty, deleting and recreating it if necessary.

    Args:
        folder (str): Path to the folder to reset or create.

    Returns:
        None
    """
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)


def print_video_fps(video_folder: str) -> None:
    """
    Prints the frames per second (FPS) for each video file in a folder.

    Args:
        video_folder (str): Path to the folder containing video files.
    """
    for video_file in os.listdir(video_folder):
        if video_file.endswith('.avi'):
            video_path = os.path.join(video_folder, video_file)
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            logging.info(f'Video: {video_file}, FPS: {fps}')
            cap.release()


def extract_frames_from_video(video_path: str, output_folder: str, frame_step: int) -> None:
    """
    Extracts frames from a single video at a specified interval.

    Args:
        video_path (str): Path to the video file.
        output_folder (str): Path to the folder where extracted frames will be saved.
        frame_step (int): Step size for frame extraction (e.g., every nth frame).
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if success and frame_count % frame_step == 0:
            frame_filename = f'frame_{frame_count}.jpg'
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    logging.info(f'Extracted frames for {video_path} into {output_folder}')


def extract_frames(video_folder: str, output_folder: str, frame_step: int) -> None:
    """
    Extracts frames from all videos in a folder at a specified interval.

    Args:
        video_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder where extracted frames will be saved.
        frame_step (int): Step size for frame extraction (e.g., every nth frame).
    """
    ensure_folder_exists(output_folder)

    for video_file in os.listdir(video_folder):
        if video_file.endswith('.avi'):
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]
            video_output_folder = os.path.join(output_folder, video_name)

            ensure_folder_exists(video_output_folder)
            extract_frames_from_video(video_path, video_output_folder, frame_step)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='{message}', style='{')
    parser = argparse.ArgumentParser(description='Extract frames from videos in a folder.')
    parser.add_argument('--video_folder', required=True, help='Path to the folder containing video files.')
    parser.add_argument('--output_folder', required=True, help='Path to the folder where frames will be saved.')
    parser.add_argument('--frame_step', type=int, help='Step size for frame extraction.')
    args = parser.parse_args()

    ensure_folder_exists(args.output_folder)
    print_video_fps(args.video_folder)
    extract_frames(args.video_folder, args.output_folder, args.frame_step)
