import subprocess
import sys
sys.path.append("/home/weihanx/videogpt/workspace/UniVTG")
import soundfile as sf
import utils
import librosa
import pathlib
import numpy as np
import pandas as pd
from pathlib import Path
import audio_preprocess
from slicer2 import Slicer
from audio_preprocess import extract_audio_wav_ffmpeg, slicer, separate_sf_song
import json
from config import PROMPT_TYPE, STORY_LIKE, NUM_CLIPS, THRESHOLD_LIST, NOISE_THRES, FPS, MODEL_VERSION, OUTPUT_FEAT_SIZE
import os
import logging
import pdb
import time
import torch
import argparse
import subprocess

from moviepy.editor import VideoFileClip

import cv2
import json
from helper import load_json, save_json, load_txt, save_txt
import logging
import torch.backends.cudnn as cudnn
from main.config import TestOptions, setup_model
from utils.basic_utils import l2_normalize_np_array
import matplotlib.pyplot as plt
from run_on_video import clip, vid2clip, txt2clip

from openai import OpenAI
import torch
from TTS.api import TTS

############### Default Settings ################
client = OpenAI(
    # This is the default and can be omitted
    api_key=Your_API_KEY,
)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model 	CLIP-B/16: QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo
parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='/home/weihanx/videogpt/deepx_data6/demo/demo_tf/feature')
parser.add_argument('--resume', type=str, default='./results/omni/model_best.ckpt')
parser.add_argument("--log_path", type=str, default="/home/weihanx/videogpt/deepx_data6/demo/demo_tf/demo_logging.txt", help="Path to the log file")

args = parser.parse_args()


clip_model, _ = clip.load(MODEL_VERSION, device, jit=False)


logging.basicConfig(
    filename="/home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_tf/demo_logging.txt",  # Use the logging file path from command-line argument
    filemode='a',  # Append mode
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Log message format
)
logger = logging.getLogger('MyLogger')
logger.info(f"Training free demo with GPT4 narration. Using binary search to find the threshold.")

logger.info(f"Save feature at = {args.save_dir}")
logger.info(f"Resume from = {args.resume}")
logger.info(f"clip_model = {MODEL_VERSION}")
logger.info(f"device = {device}")




################## Video Processing ######################
def load_json(filename):
    """Load data from a JSON file."""
    with open(filename, encoding="utf8") as f:
        return json.load(f)

def get_video_duration(file_path):
    video = VideoFileClip(str(file_path))
    duration = video.duration  # Duration in seconds
    return duration

def extract_and_concat_segments(video_path, intervals, output_path, segment_duration=None, fps=1):
    # Temporary file list for storing segment file names
    filelist_path = "filelist.txt"
    with open(filelist_path, "w") as filelist:
        for i, (start_time, end_time) in enumerate(intervals): # might have multiple intervals in this clip
            segment_file = f"segment_{i}.mp4"
            # Calculate duration if not explicitly provided 
            assert fps == 1  # for this project, it is 1
            duration = str(end_time - start_time + 1 / fps) if segment_duration is None else segment_duration
            print(f"start time = {start_time}, end time = {end_time}, duration = {duration}")  # correct!
            # FFmpeg command to extract a segment from the video
            extract_command = [
                'ffmpeg',
                '-ss', str(start_time),      # Start time of the segment
                '-t', duration,              # Duration of the segment
                '-i', video_path,            # Input video file
                '-map', '0',                 # Include all streams (video and audio)
                '-c:v', 'libx264',           # Re-encode video with x264
                '-c:a', 'aac',               # Re-encode audio with AAC
                '-avoid_negative_ts', 'make_zero',  # Avoid negative timestamps
                segment_file                 # Output segment file
            ]
            subprocess.run(extract_command, check=True)
            # Write the file name to the filelist for concatenation
            filelist.write(f"file '{segment_file}'\n")

    # FFmpeg command to concatenate all segments
    concat_command = [
        'ffmpeg',
        '-f', 'concat',               # Use the concat demuxer
        '-safe', '0',                 # Allow unsafe file paths
        '-i', filelist_path,          # Input file list
        '-c', 'copy',                 # Copy codecs without re-encoding
        output_path                   # Path for the output file
    ]
    subprocess.run(concat_command, check=True)

    # Optional: clean up segment files
    for i in range(len(intervals)):
        os.remove(f"segment_{i}.mp4")


def merge_audio_video(video_path, audio_path, output_path):
    # Command to run FFmpeg: combine video and audio with the audio overriding any existing audio stream in the video
    command = [
        'ffmpeg',
        '-i', video_path,  # Input video path
        '-i', audio_path,  # Input audio path
        '-c:v', 'copy',    # Copy video codec settings (no re-encoding)
        '-c:a', 'aac',     # Specify audio codec (ensure compatibility)
        '-strict', 'experimental',  # Sometimes needed if using aac
        '-map', '0:v:0',   # Map video stream from the video input
        '-map', '1:a:0',   # Map audio stream from the audio input
        output_path       # Path for the output file
    ]
    # Execute the FFmpeg command
    subprocess.run(command, check=True)

# replace the track
def replace_audio(video_input, audio_input, output_video):
    # Command to run FFmpeg: remove existing audio and add new audio
    command = [
        'ffmpeg',
        '-i', video_input,    # Input video file
        '-i', audio_input,    # New audio file
        '-c:v', 'copy',       # Copy video codec settings (no re-encoding)
        '-c:a', 'aac',        # Specify audio codec (ensure compatibility)
        '-strict', 'experimental',  # Sometimes needed if using AAC
        '-map', '0:v:0',      # Map video stream from the video input
        '-map', '1:a:0',      # Map audio stream from the audio input
        output_video          # Path for the output file
    ]
    # Execute the FFmpeg command
    subprocess.run(command, check=True)

# def merge_videos(video_files, output_path):
#     # Write the file paths to a temporary text file
#     with open("concat_file.txt", "w") as f:
#         for file in video_files:
#             f.write(f"file '{file}'\n")
#     command = [
#         'ffmpeg',
#         '-f', 'concat',              # Use the concat demuxer
#         '-safe', '0',                # Allow unsafe file paths
#         '-i', 'concat_file.txt',     # Input list file
#         '-c:v', 'libx264',           # Re-encode video using x264 codec
#         '-preset', 'fast',           # Faster encoding with less compression
#         '-crf', '22',                # Constant Rate Factor for quality
#         '-c:a', 'aac',               # AAC audio codec
#         '-b:a', '192k',              # Audio bitrate
#         output_path                  # Output file path
#     ]
#     subprocess.run(command, check=True)

def merge_videos(video_files, output_path):
    # Write the file paths to a temporary text file
    with open("concat_file.txt", "w") as f:
        for file in video_files:
            f.write(f"file '{file}'\n")
    
    command = [
        'ffmpeg',
        '-f', 'concat',              # Use the concat demuxer
        '-safe', '0',                # Allow unsafe file paths
        '-i', 'concat_file.txt',     # Input list file
        '-c', 'copy',                # Copy video and audio streams (no re-encoding)
        output_path                  # Output file path
    ]
    
    subprocess.run(command, check=True)
# get time intervals

def load_model():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse(args)
    # pdb.set_trace()
    cudnn.benchmark = True
    cudnn.deterministic = False

    if opt.lr_warmup > 0:
        total_steps = opt.n_epoch
        warmup_steps = opt.lr_warmup if opt.lr_warmup > 1 else int(opt.lr_warmup * total_steps)
        opt.lr_warmup = [warmup_steps, total_steps]

    model, criterion, _, _ = setup_model(opt)
    return model

vtg_model = load_model()

def convert_to_hms(seconds):
    return time.strftime('%H:%M:%S', time.gmtime(seconds))

def load_data(txt, vid_path, FPS):
    # print(f"vid path = {vid_path}, txt = {txt}")
    # torch.Size([1, 250, 514]), src_txt = torch.Size([1, 32, 512])
    vid = extract_vid(vid_path, FPS)
    txt = extract_txt(txt)
    # print("vid = ", vid.shape, "txt = ", txt.shape)
    vid = vid.astype(np.float32)
    txt = txt.astype(np.float32)

    vid = torch.from_numpy(l2_normalize_np_array(vid))
    txt = torch.from_numpy(l2_normalize_np_array(txt))
    clip_len = 1 / FPS

    ctx_l = vid.shape[0]

    timestamp =  ( (torch.arange(0, ctx_l) + clip_len / 2) / ctx_l).unsqueeze(1).repeat(1, 2)

    if True:
        tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
        tef_ed = tef_st + 1.0 / ctx_l
        tef = torch.stack([tef_st, tef_ed], dim=1)  # (Lv, 2)
        vid = torch.cat([vid, tef], dim=1)  # (Lv, Dv+2)

    src_vid = vid.unsqueeze(0).to(device)
    src_txt = txt.unsqueeze(0).to(device)
    src_vid_mask = torch.ones(src_vid.shape[0], src_vid.shape[1]).to(device)
    src_txt_mask = torch.ones(src_txt.shape[0], src_txt.shape[1]).to(device)

    return src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l

def forward(model, txt, vid_path, FPS):

    clip_len = 1 / FPS
    src_vid, src_txt, src_vid_mask, src_txt_mask, timestamp, ctx_l = load_data(txt, vid_path, FPS
    )
    src_vid = src_vid.to(device)
    src_txt = src_txt.to(device)
    src_vid_mask = src_vid_mask.to(device)
    src_txt_mask = src_txt_mask.to(device)

    model.eval()
    with torch.no_grad():
        output = model(src_vid=src_vid, src_txt=src_txt, src_vid_mask=src_vid_mask, src_txt_mask=src_txt_mask)
    # print(f"outsrc_vid = {src_vid.shape}, src_txt = {src_txt.shape}")
    # prepare the model prediction
    pred_logits = output['pred_logits'][0].cpu().T # f --> 1,150
    pred_spans = output['pred_spans'][0].cpu() # b
    pred_saliency = output['saliency_scores'].cpu() # s

    saliency_score = pred_saliency.numpy()

    logits = pred_logits.numpy()
    # print(f"shape of saliency score = {saliency_score.shape}, shape of logits = {logits.shape}")
    highlight_score = saliency_score + logits
    return highlight_score
    
def extract_vid(vid_path, FPS):

    clip_len = 1/FPS
    video_name = vid_path.parts[-2]
    clip_name = vid_path.parts[-1].split('.')[0] # only need clip_3
    vid_path = str(vid_path)
    save_path = Path("/home/weihanx/videogpt/deepx_data6/demo/video_clip_feature") / video_name / clip_name
    save_path.mkdir(parents=True, exist_ok=True)
    check_exist = save_path / "vid.npz"
    if not check_exist.exists():
        vid_features = vid2clip(clip_model, vid_path, str(save_path), clip_len=clip_len)

    else:
        vid_features = np.load(check_exist)['features']
    return vid_features

def extract_txt(txt):
    txt_features = txt2clip(clip_model, txt, args.save_dir) # directly returned, not loading
    return txt_features

def load_mp4_files(folder_path):
    # Convert the folder path to a Path object
    folder = Path(folder_path)
    
    # Use the glob method to find all .mp4 files in the specified folder
    mp4_files = list(folder.glob('*.mp4'))
    return mp4_files

def get_saliency_curve(text, video_name, video_clip_folder):
    """
    Given the text and video name, extract the saliency curve for the video
    """
    mp4_folder_path = video_clip_folder / video_name[0] / video_name
    mp4_files = load_mp4_files(mp4_folder_path)
    saliency_score_concat = []
    for mp4 in mp4_files: # all the path name
        # what is saved is current vid features
        saliency_score = forward(vtg_model, text, mp4, FPS)
        saliency_score = saliency_score.squeeze() # should be 1 d array
        saliency_score_concat.append(saliency_score)
        # （1，250) numpy array

    saliency_score_concat = np.concatenate(saliency_score_concat) 
    # should be 10*250

    return saliency_score_concat



def find_consecutive_intervals(indices):
    if len(indices) == 0:
        return []

    intervals = []
    start = indices[0]
    
    for i in range(1, len(indices)):
        # If the current index is not consecutive, close the previous interval
        if indices[i] != indices[i - 1] + 1:
            intervals.append([start, indices[i - 1]])
            start = indices[i]  # Start a new interval
    
    # Add the last interval
    intervals.append([start, indices[-1]])
    
    return intervals

def filter_intervals_by_duration(intervals, min_duration):
    return [interval for interval in intervals if (interval[1] - interval[0] + 1) >= min_duration]


# text, video_name, input_dir, mid, FPS, last, last_last, overlap_bound, min_duration
def clip_interval_extraction(text, video_name, video_clip_folder, threshold, FPS, last, last_last, min_dur, tolerance):
    clip_sal_score = get_saliency_curve(text, video_name, video_clip_folder)

# text, video_name, input_dir, best_threshold, FPS, global_track
    # print(f"clip_sal_score = {clip_sal_score.shape}")
    clip_sal_score = clip_sal_score.flatten()  # Ensure it's a 1D array
    save_arr_path = Path("/home/weihanx/videogpt/deepx_data6/demo/demo_tf_0924") / video_name[0] / video_name / f"sal_curve_{str(text)[:5]}.npy"
    save_arr_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_arr_path.exists():
        np.save(save_arr_path, clip_sal_score)
    global_track = np.ones(clip_sal_score.shape[0])  # remove those should not be selected
    for ts in last:
        start, end = ts
        start = start + tolerance
        end = end - tolerance # should be include
        global_track[start:end+1] = 0
    for ts in last_last:
        start, end = ts
        start = start + tolerance
        end = end - tolerance
        global_track[start:end+1] = 0

    available_indices = np.where(global_track == 1)[0]
    # print(f"available_indices = {available_indices}")
    # Step 2: Apply the threshold (clip_score > 0.4) to these indices
    thresholded_indices = available_indices[clip_sal_score[available_indices] > threshold]
    # print(f"threshold_indices = {thresholded_indices}")
    # print(f"clip_sal_score = {clip_sal_score[556:563]}")
    thresholded_indices = thresholded_indices.tolist()
    consective_intervals = find_consecutive_intervals(thresholded_indices)
    filtered_interval = filter_intervals_by_duration(consective_intervals, min_dur)
    total_len = sum((interval[1] - interval[0] + 1) for interval in filtered_interval)
    return total_len, filtered_interval

def gen_tts(script, out_speech_dir, video_name, chunk_num):
    # Caution! Chunk number is the video id number in the narration
    out_path = out_speech_dir / video_name[0] / video_name / f"chunk_{chunk_num}_speech.wav"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        # tts.tts_to_file(text=joined_sentence, file_path=str(out_path))
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=script,
        )
        response.stream_to_file(out_path)

    else:
        print(f"Audio file already exists: {out_path}")
    audio, sample_rate = librosa.load(str(out_path), sr=None)
    duration = librosa.get_duration(y=audio, sr=sample_rate)

    return duration


def get_duration_csv(video_name, narr_base_dir, out_speech_dir, time_duration_path):
    # get time interval from csv
    # 
    csv_path = narr_base_dir /  video_name[0] / video_name / "ts_comb.csv"
    df = pd.read_csv(csv_path)
    time_intervals = {}
    for index, row in df.iterrows():
        script = row["text"]
        id = row['id'] # index by id column
        duration = gen_tts(script, out_speech_dir, video_name, id)
        time_intervals[id] = duration  # check id = duration
    save_json( time_duration_path, time_intervals)

def binary_search_threshold(text, video_name, input_dir, FPS, text_duration, last, last_last, min_duration, min_threshold, max_threshold, tolerance, precision=0.001):
    """
    text, video_name, input_dir, FPS, text_duration, overlap_bound, last, last_last, min_duration, min_threshold=-1, max_threshold=1.0, precision=0.001
    Use binary search to find the threshold such that the total_duration is larger than text_duration and as close as possible to text_duration.
    """
    low, high = min_threshold, max_threshold
    best_threshold = None
    best_duration = float('inf')
    best_time_intervals = None

    while high - low > precision:
        mid = (low + high) / 2

        total_duration, _= clip_interval_extraction(text, video_name, input_dir, mid, FPS, last, last_last, min_duration, tolerance)

        if total_duration > text_duration:
            if total_duration < best_duration:
                
                best_duration = total_duration
                best_threshold = mid
            low = mid  # Move down to find a closer value
        else:
            high = mid  # Move up to find a closer value

    best_duration, best_time_intervals= clip_interval_extraction(text, video_name, input_dir, best_threshold, FPS, last, last_last, min_duration, tolerance)

    return best_duration, best_time_intervals
    
def get_time_interval(input_dir, output_dir, narr_folder, video_name, FPS, speech_save_folder, min_duration, tolerance):
    clip_duration_dict = {}
    narr_scripts = narr_folder / video_name[0] / video_name / "ts_comb.csv"
    narr_scripts = pd.read_csv(narr_scripts)
    duration_len = speech_save_folder / video_name[0] / video_name / f"speech_duration.json"
    duration_len_dict = load_json(duration_len)
    # global selected intervals: should get the total number of frame:
    # frame folder:
    body_path = Path("/home/weihanx/videogpt/data_deepx/documentary/sliced_video") / video_name[0] / video_name / "main.mp4"
    body_path_duration = get_video_duration(body_path)


    last_last = []
    last = []
    min_threshold = -1
    max_threshold = 2
    for row, index in narr_scripts.iterrows():
        text = index['text']
        id = index['id']
        text_duration = duration_len_dict[str(index['id'])]
        text_duration = text_duration + 1# 1 second buffer time: keep consistent

        best_duration, time_intervals = binary_search_threshold(text, video_name, input_dir, FPS, text_duration, last, last_last, min_duration, min_threshold, max_threshold, tolerance, precision=0.000001)
        last_last = last
        last = time_intervals 
        clip_duration_dict[id] = time_intervals
    # save the clip duration dict
    output_json_path = output_dir / video_name[0] / video_name / f"time_interval_min_{min_duration}.json"
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(output_json_path, clip_duration_dict)


if __name__ == "__main__":
    pass
        