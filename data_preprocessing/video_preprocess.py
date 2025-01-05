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
from utils import load_txt, save_json, save_txt
"""
Split the video into intro and main
"""
@utils.resolve_paths
def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Youtube Video Download"
    )
    parser.add_argument("-p", "--playlist", type=pathlib.Path, help="playlist names")
    parser.add_argument("-n", "--names", type=pathlib.Path, help="input names")
    parser.add_argument(
        "-i", "--in_dir", type=pathlib.Path, help="input data directory"
    )
    parser.add_argument(
        "-o", "--out_dir", type=pathlib.Path, help="output directory"
    )
    parser.add_argument(
        "-m", "--meta_dir", type=pathlib.Path, help="metadata out directory"
    )
    parser.add_argument(
        "-e",
        "--ignore_exceptions",
        action="store_true",
        help="whether to ignore all exceptions",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=1,
        help="number of jobs",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="show warnings only"
    )
    return parser.parse_args(args=args, namespace=namespace)

def convert_to_seconds(time_str):
    # Split the string into minutes and seconds
    minutes, seconds = map(int, time_str.split(':'))
    
    # Calculate total seconds
    total_seconds = minutes * 60 + seconds
    return total_seconds

def split_video_ffmpeg(video_path, split_point, output_folder, video_name):
    # Convert "mm:ss" format to seconds
    split_point_sec = convert_to_seconds(split_point)
    print(f"Split point in seconds: {split_point_sec}")

    parent_dir = output_folder / video_name[0] / video_name 
    parent_dir.mkdir(parents=True, exist_ok=True)
    intro_path = parent_dir / "intro.mp4"
    main_path = parent_dir / "main.mp4"
    command_intro = [
        'ffmpeg',
        '-i', video_path,
        '-t', str(split_point_sec),  # Set the duration of the intro part
        '-c', 'copy',
        intro_path
    ]

    # Command to extract the second part (from the split point to the end)
    command_main = [
        'ffmpeg',
        '-i', video_path,
        '-ss', str(split_point_sec),  # Set the start time of the main part
        '-c', 'copy',
        main_path
    ]

    # Execute the commands
    try:
        # Run the command to create the intro part
        subprocess.run(command_intro, check=True)
        # Run the command to create the main part
        subprocess.run(command_main, check=True)
    except subprocess.CalledProcessError as e:
        print("Failed to split the video:", e)


def prepare_path(old_csv_path, new_csv_path):
    df = pd.read_csv(old_csv_path)
    df['Video_Name'] = df['URL'].apply(lambda x: x.split('=')[-1])
    df.to_csv(new_csv_path, index=False, header=True)

def find_title_in_video(video_path, title):
    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    found = False

    while not found and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_number += 1
            # Convert frame to format suitable for OCR (optional, depending on the video quality)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            image = Image.fromarray(gray)

            # Perform OCR
            text = pytesseract.image_to_string(image)
            
            # Check if title is in OCR output
            if title.lower() in text.lower():
                found = True
                print(f"Title '{title}' found in frame {frame_number}!")
            else:
                print(f"Title not found in frame {frame_number}.")
        else:
            print("End of video or read error.")
            break

    cap.release()

def video_split_scale(csv_path, input_dir, out_dir, file_path):
    
    """
    csv_path: csv with videoname and split point
    input_dir: video directory
    out_dir: processed_video
    """
    df = pd.read_csv(csv_path)
    valid_file_name = load_txt(file_path)
    for index, row in df.iterrows():
        if row['Video_Name'] not in valid_file_name:
            continue
        video_name = row['Video_Name']  # Make sure this is the correct column for the video file path
        video_filename = f"{video_name}.mp4"
        # video_path, split_point, output_folder, video_name
        video_path = input_dir / video_name[0] / f'{video_name}.mp4'
        time = str(row['Time'])  # Assuming 'Time' column has the split time in minutes
 
        split_video_ffmpeg(video_path, time, out_dir , video_name)



def save_frames(video_path, output_folder):
    """
    Saves frames from the video at a rate of 1 frame per second.

    :param video_path: Path to the video file.
    :param output_folder: Folder where the extracted frames should be saved.
    """
    command = [
        'ffmpeg',
        '-i', video_path,                  # Input video file
        '-vf', 'fps=1',                    # Set frame rate for output
        f'{output_folder}/frame_%04d.jpg'  # Output path pattern
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    pass
