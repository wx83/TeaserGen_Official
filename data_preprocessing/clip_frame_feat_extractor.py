# This is used for preparing dataset
# loop through training data
import json
from pathlib import Path
import torch
from utils import save_json, load_json, save_txt, load_txt
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
import os
import torch
from collections import defaultdict
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop, ToTensor
from transformers import RobertaModel, RobertaTokenizer
import torch
import pandas as pd
import pathlib
from pathlib import Path
import subprocess   
import whisper_timestamped as whisper
import json
import pathlib
import numpy as np
import math
import pandas as pd
from pathlib import Path
from utils import load_txt, load_json
import logging
import ffmpeg
# Find command: find . -type f -name "*.csv" | sed 's/\.csv$//'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






def save_json(filename, data):
    """Save data as a JSON file."""
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f)

def combine_segments(segments, threshold=0.5):
    combined_segments = []
    current_segment = segments[0]
    
    for segment in segments[1:]:
        if abs(current_segment['end'] - segment['start']) < threshold:
            current_segment['end'] = segment['end']
            current_segment['text'] += " " + segment['text']
        else:
            combined_segments.append(current_segment)
            current_segment = segment
            
    combined_segments.append(current_segment)
    return combined_segments



# break
# print(json.dumps(result, indent = 2, ensure_ascii = False))
def extract_audio_wav_ffmpeg(input_video_path, output_audio_path):
    """
    Extracts the audio from an MP4 video file using ffmpeg and saves it as a WAV file.
    
    Args:
        input_video_path (str): The path to the input MP4 video file.
        output_audio_path (str): The path where the output WAV audio file will be saved.
    """
    try:
        # Execute the ffmpeg command to extract audio
        ffmpeg.input(str(input_video_path)).output(str(output_audio_path), acodec='pcm_s16le', ar='44100').run()
    except ffmpeg.Error as e:
        # Print the stderr output from ffmpeg for debugging
        print("ffmpeg failed with stderr:", e.stderr.decode('utf8'))
        raise RuntimeError("Failed to extract audio with ffmpeg") from e


def get_clip_audio_script(input_dir, output_audio_dir, output_script_dir, video_frame_list_path, model):
    video_frame_list = load_txt(video_frame_list_path)
    for vn in video_frame_list:
        # loop through all mp4 files
        print(f"vn = {vn}")
        video_input_dir = input_dir / vn[0] / vn # clip
        for video_path in video_input_dir.rglob("*.mp4"): # clip 1, 2, 3, 4
            df = pd.DataFrame(columns=["id", "start", "end", "text"]) # each clip has one dataframe
            clip_name = video_path.stem
            # print(f"clip_name = {clip_name}")
            audio_path = output_audio_dir / vn[0] / vn / f"{clip_name}.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            if not audio_path.exists():
                extract_audio_wav_ffmpeg(video_path, audio_path)
            audio = whisper.load_audio(audio_path)
            result = whisper.transcribe(model, audio, vad=True, detect_disfluencies=True,beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), language="en")
            # print(f"result = {result.keys()}")
            # save_json_path = output_script_dir / vn[0] / vn / f"{clip_name}_script.json"
            # save_json_path.parent.mkdir(parents=True, exist_ok=True)
            # result = save_json(save_json_path,result)
            text_list = result["segments"]
            combined_text_list = combine_segments(text_list)
            for t in combined_text_list:

                new_row = pd.DataFrame([[t["id"], t["start"], t["end"], t["text"]]], columns=["id", "start", "end", "text"])
                df = pd.concat([df, new_row], ignore_index=True)
            output_script_dir.mkdir(parents=True, exist_ok=True)
            save_csv = output_script_dir / vn[0] / f"{vn}_{clip_name}.csv"
            save_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(save_csv)
        # break
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
        f'{output_folder}/frame_%05d.jpg'  # Output path pattern
    ]
    subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def video_to_frame(video_dir, video_name_path, output_folder):
    """
    Saves frames from the video at a rate of 1 frame per second.

    :param video_path: Path to the video file.
    :param output_folder: Folder where the extracted frames should be saved.
    #  video_name_path: j1Pb8YU8wQo_clip_4
    """
    for i in range(10):
        load_video = video_dir / video_name_path[0] / video_name_path / f"clip_{i}.mp4" 
        print(f"processing... = {load_video}")
        save_dir = output_folder / video_name_path[0] / f"{video_name_path}_clip_{i}"
        save_dir.mkdir(parents=True, exist_ok=True)

        save_frames(str(load_video), save_dir)
        print(f"finished... = {video_name_path}")



        
        # get frame
        # get feature
# step 2:
# cut all video clips into frames with fps = 1
# generate video features, fps =1, use clip to get features, should be number of clips * 512
def load_frames_from_video(video_dir):
    """Load frames from a given video directory."""
    frame_files = sorted([f for f in video_dir.iterdir() if f.is_file()])
    frame_files = frame_files[::3]  # Skip every other frame to reduce the number of frames: prop
    frames = []
    for f in frame_files:
        print(f"frame = {f}")
        frame = Image.open(video_dir / f)
        frames.append(frame)
    return frames

def apply_clip_to_frames(video_dir, model, processor, device="cuda"):
    embeddings = []
    model = model.to(device)
    frame_files = sorted([f for f in video_dir.iterdir() if f.is_file()])
    for frame in frame_files:
        frame_path = video_dir / frame
        frame_tensor = Image.open(frame_path)
        inputs = processor(images=frame_tensor, return_tensors="pt").to(device)  # Process the frame with processor
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)  # batch_size x 512
            embedding = image_features.squeeze(0).cpu().numpy()  # Remove batch dimension
        embeddings.append(embedding)
    
    return np.array(embeddings)

def save_embeddings(embeddings, output_path):
    """Save embeddings to a specified file."""
    np.save(output_path, embeddings)

def generate_image_emb_b32(input_dir, video_name_path, output_dir, device = "cuda"):
    video_name_list = load_txt(video_name_path) 
    # Load the pre-trained ResNet model
    fail_dataset = []
    # model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    # processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    for video_name in video_name_list:

        for idx in range(10):
            video_dir = input_dir / video_name[0] / f"{video_name}_clip_{idx}"
            frame_folder = load_frames_from_video(video_dir)

            embeddings = apply_clip_to_frames(frame_folder, model,processor)
            print(f"clip = {idx}, shape of embeddings = {embeddings.shape}")
        # Save the embeddings to the output directory 
            output_path = output_dir /  video_name[0] / f"{video_name}_clip_{idx}" / f"{video_name}_clip_{idx}.npy"
            output_path.parent.mkdir(parents=True, exist_ok=True)
        # output_path = os.path.join(output_dir, f"{video_name}_embeddings.npy")
        
            save_embeddings(embeddings, output_path)
        
        # break
    output_path = output_dir / "fail_dataset.txt"
    with open(output_path, "w") as f:
        for item in fail_dataset:
            f.write("%s\n" % item)

def generate_image_emb_l14(input_dir, video_name_path, output_dir, device = "cuda"):
    video_name_list = load_txt(video_name_path) 

    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")
    model.eval()
    for video_name in video_name_list:
        print(f"processing... = {video_name}")
        video_dir = input_dir / video_name[0] / video_name
        # frame_folder = load_frames_from_video(video_dir) # probably need to load separately
        # print(f"frame_folder = {frame_folder}")
        embeddings = apply_clip_to_frames(video_dir, model, processor)
        output_path = output_dir / f"{video_name}_clip.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
    # output_path = os.path.join(output_dir, f"{video_name}_embeddings.npy")



if __name__ == "__main__":
    video_name_list = Path("/home/weihanx/videogpt/workspace/UniVTG/error_file.txt")
    input_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/body_frame_train")
    output_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_frame_768_emb_main")
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_image_emb_l14(input_dir, video_name_list, output_dir, device = "cuda:0")
    # filepath = "/home/weihanx/videogpt/data_deepx/documentary/body_frame_train/W/WepnzaNLLMI/frame_01017.jpg"
    # frame = Image.open(filepath)
    # print(frame.size)
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset")
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/demo_eval/actual_intro/matched_array_ed")
    # test_video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # prepare_annotation(input_dir, output_dir, test_video_name_path)
    # video_frame_list = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # model = whisper.load_model("tiny", device="cuda")
    # output_script_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/transcript")
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/demo/video_clip")
    # output_audio_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_clip_audio")
    # get_clip_audio_script(input_dir, output_audio_dir, output_script_dir, video_frame_list, model)
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset")
    # input_csv_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/transcript")
    # video_clip_folder = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip")
    # video_frame_folder = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip_frame")
    # # video_name_path_list_path = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/add.txt")
    # # construct_feature_per_docu(video_name_path_list_path, input_csv_dir, video_clip_folder, output_dir)
    # video_name_list = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_clip_frame_512_emb")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # # for v in video_name:
    # generate_image_emb_b32(input_dir, video_name_list, output_dir, device = "cuda")
    # #     if v != "j1Pb8YU8wQo_clip_4":
    #         continue
     
    # for i in video_name:
    #     print(f"processing... = {i}")
    #     video_to_frame(video_clip_folder, i, video_frame_folder)