import cv2
import pytesseract
from PIL import Image
import os
import numpy as np
import pandas as pd
import subprocess
from moviepy.editor import VideoFileClip
import ffmpeg
import utils
import argparse
import pathlib
from pathlib import Path


def save_frames(video_path, output_folder):
    """
    Saves frames from the video at a rate of 1 frame per second.

    :param video_path: Path to the video file.
    :param output_folder: Folder where the extracted frames should be saved.
    """
    command = [
        'ffmpeg',
        '-i', video_path,                  # Input video file
        '-vf', 'fps=1/2',                    # Set frame rate for output
        f'{output_folder}/frame_%05d.jpg'  # Output path pattern
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    video_name_list = utils.load_txt("/home/weihanx/videogpt/workspace/start_code/eval/train_val_files.txt")
    input_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/sliced_video")

    parent_dir = Path("/home/weihanx/videogpt/deepx_data7/frame_outputs_intro_test_fpshalf")
    parent_dir.mkdir(parents=True, exist_ok=True)
    for video_name in video_name_list:
        print(f"processing... = {video_name}")
        load_video = input_dir / video_name[0] / video_name / f"intro.mp4"
        save_dir = parent_dir / video_name[0] / video_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_frames(str(load_video), save_dir)
# not fialed?????
