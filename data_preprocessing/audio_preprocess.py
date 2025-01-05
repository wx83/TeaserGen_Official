import soundfile as sf
import numpy as np
from ensemble import EnsembleNet  # Ensure you have the correct import path
import os
import pathlib
from helper import load_txt, save_txt, load_json, save_json
from pathlib import Path
import pandas as pd
import ffmpeg
import torch
import os
import time
from slicer2 import Slicer
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
    
device = "cuda:0"
ensemble_net = EnsembleNet()

if torch.cuda.is_available():
    print("CUDA is available. GPU can be used.")
else:
    print("CUDA is not available. Running on CPU.")
sr = 44100
# RMS (root mean score) to measure the quiteness of the audio and detect silent parts
slicer = Slicer(
    sr=sr,
    threshold=-25,
    min_length=3000, # at least 5 seconds per slice, sfx?
    min_interval=300,
    hop_size=10, # Length of each RMS frame, presented in milliseconds: 
    max_sil_kept=2000 # The maximum silence length kept around the sliced audio, 1 second, if it is less than 2 second, I will treat them continuous
)

"""
This file is used to deal with sound
1. separate soundtrack from videos
2. separate the soundtrack into dialog, sound effect and music
3. For long audio tracks, slice them and use overlap range to combine
"""

def extract_audio_wav_ffmpeg(input_video_path, output_audio_path):
    """
    Extracts the audio from an MP4 video file using ffmpeg and saves it as a WAV file.
    
    Args:
        input_video_path (str): The path to the input MP4 video file.
        output_audio_path (str): The path where the output WAV audio file will be saved.
    """
    try:

        ffmpeg.input(str(input_video_path)).output(str(output_audio_path), acodec='pcm_s16le', ar='44100').run()
    except ffmpeg.Error as e:

        print("ffmpeg failed with stderr:", e.stderr.decode('utf8'))
        raise RuntimeError("Failed to extract audio with ffmpeg") from e

def audio_extract_scale(video_name_path, input_dir, out_dir, part="intro"):
    """
    input dir: processed_video
    out dir: audio
    """
    video_urls = load_txt(video_name_path)
    for video_name in video_urls:
        # os.makedirs(out_dir, exist_ok=True)
        video_path = input_dir / video_name[0] / video_name / f"{part}.mp4"
        parent_dir = out_dir / video_name[0] / video_name
        parent_dir.mkdir(parents=True, exist_ok=True)  
        audio_path =  parent_dir / video_name[0] / video_name / f"{part}.wav"

        print(f"audio_path = {audio_path}")
        try:
            extract_audio_wav_ffmpeg(video_path, audio_path)
        except:
            print(f"video_path = {video_path} cannot found")

def audio_extract_scale_zeroshot(video_name_path, input_dir, out_dir):
    """
    input dir: processed_video
    out dir: audio
    """
    video_urls = load_txt(video_name_path)
    for video_name in video_urls:

        video_path = input_dir / f"{video_name}.mp4"

        audio_path =  out_dir / f"{video_name}.wav"
        print(f"audio_path = {audio_path}")
        try:
            extract_audio_wav_ffmpeg(video_path, audio_path)
        except:
            print(f"video_path = {video_path} cannot found")


def split_long_slices(audio_data, max_duration, sample_rate):
    """Splits audio data if longer than max_duration seconds."""
    max_samples = max_duration * sample_rate
    if len(audio_data) <= max_samples:
        return [audio_data]
    return [audio_data[i:i + max_samples] for i in range(0, len(audio_data), max_samples)]



def smooth_transitions(audio_arrays):
    """Apply a simple linear fade in and fade out to each chunk for multi-channel audio."""
    if not audio_arrays:
        return audio_arrays
    window_size = int(0.05 * 44100)  # 50 ms fade
    for i in range(1, len(audio_arrays)):
        if len(audio_arrays[i-1]) >= window_size and len(audio_arrays[i]) >= window_size:
            fade_out = np.linspace(1, 0, window_size)[:, np.newaxis]
            fade_in = np.linspace(0, 1, window_size)[:, np.newaxis]
            overlap = audio_arrays[i-1][-window_size:] * fade_out + audio_arrays[i][:window_size] * fade_in
            audio_arrays[i-1][-window_size:] = audio_arrays[i-1][-window_size:] * fade_out
            audio_arrays[i][:window_size] = overlap
    return audio_arrays

def process_audio_chunk(chunk_data, sample_rate):
    separated_audio, sample_rates = None, None
    try:
        # Thread-safe separation assumed
        separated_audio, sample_rates = ensemble_net.separate_music_file(chunk_data, sample_rate)

    except Exception as e:
        print(f"Error processing chunk: {e}")
        torch.cuda.empty_cache()  # Clear cache on error as well
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return separated_audio, sample_rates


def process_audio_file(file_path, slicer, max_slice_duration=60):
    """Process an audio file, ensuring no slice exceeds max_slice_duration seconds."""
    with sf.SoundFile(file_path) as audio_file:
        sample_rate = audio_file.samplerate
        audio_data = audio_file.read(dtype='float32')

    slices = slicer.slice(audio_data)
    processed_slices = {'music': [], 'dialog': [], 'effect': []}
    
    # Process each slice and further split if necessary
    for slice_data in slices:
        smaller_slices = split_long_slices(slice_data, max_slice_duration, sample_rate)
        for small_slice in smaller_slices:
            separated_audio, sample_rates = process_audio_chunk(small_slice, sample_rate)
            if separated_audio:
                for key in processed_slices.keys():
                    if key in separated_audio:
                        processed_slices[key].append(separated_audio[key])

    smoothed_tracks = {}
    for key in processed_slices.keys():
        processed_slices[key] = smooth_transitions(processed_slices[key])
        smoothed_tracks[key] = np.concatenate(processed_slices[key]) if processed_slices[key] else None

    return smoothed_tracks

def separate_sf_song(video_name,audio_path, out_dir, fig_dir, slicer, threshold, part, zero_shot):
    try:
        processed_tracks = process_audio_file(audio_path, slicer)
        for key, track in processed_tracks.items():
            if track is not None:
                if zero_shot == False:
                    output_file_path = out_dir / video_name[0] / video_name / f'{key}_{part}.wav'
                    output_file_path.mkdir(parents=True, exist_ok=True)  
                    sf.write(output_file_path, track, 44100)
                if zero_shot == True:
                    output_file_path = out_dir / f"{video_name}_{key}.wav"
                    sf.write(output_file_path, track, 44100)

                print(f"Processed {key} audio saved to {output_file_path}")
            else:
                print(f"No {key} audio processed.")


    except:
        print(f"audio_path = {audio_path} cannot be found")



def separate_sf_scale(video_name_path, input_dir, out_dir, fig_dir, slicer, threshold, part, zero_shot):
    """
    input_dir: audio track
    output_dir: audio separate folder
    """
    video_urls = load_txt(video_name_path)
    for video_name in video_urls:


        if zero_shot == False:
            parent_dir_out = out_dir / video_name[0] / video_name 
            parent_dir_out.mkdir(parents=True, exist_ok=True)
            parent_dir_fig = fig_dir / video_name[0] / video_name
            parent_dir_fig.mkdir(parents=True, exist_ok=True)    
            audio_path = input_dir / video_name[0] / video_name / f"{part}.wav"
            audio_path.mkdir(parents=True, exist_ok=True)  
            check_existence = input_dir / video_name[0] / video_name / "dialog_main.wav"
            if not check_existence.exists():
                separate_sf_song(video_name,audio_path, out_dir, fig_dir, slicer, threshold, part, zero_shot)
        else:
            audio_path = input_dir / f"{video_name}.wav"
            separate_sf_song(video_name,audio_path, out_dir, fig_dir, slicer, threshold, part, zero_shot)





def audio_extract_main(video_name_path, input_dir, out_dir, part, zeroshot):
    if torch.cuda.is_available():
        print("CUDA is available. GPU can be used.")
    else:
        print("CUDA is not available. Running on CPU.")
    if zeroshot == False:
        audio_extract_scale(video_name_path, input_dir, out_dir, part)
    else:
        audio_extract_scale_zeroshot(video_name_path, input_dir, out_dir)

if __name__ == "__main__":

    pass