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
    api_key=YOUR_API_KEY,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model 	CLIP-B/16: QVHL + Charades + NLQ + TACoS + ActivityNet + DiDeMo
parser = argparse.ArgumentParser(description='')
parser.add_argument('--save_dir', type=str, default='/home/weihanx/videogpt/deepx_data6/demo/demo_tf/feature')
parser.add_argument('--resume', type=str, default='./results/omni/model_best.ckpt')
parser.add_argument("--log_path", type=str, default="/home/weihanx/videogpt/deepx_data6/demo/demo_tf/demo_logging.txt", help="Path to the log file")

args = parser.parse_args()


clip_model, _ = clip.load(MODEL_VERSION, device, jit=False)




################## Video Processing ######################
def load_json(filename):
    """Load data from a JSON file."""
    with open(filename, encoding="utf8") as f:
        return json.load(f)

def get_video_duration(file_path):
    video = VideoFileClip(str(file_path))
    duration = video.duration  # Duration in seconds
    return duration

def load_model():
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

def get_array_saliency_score(video_name_path, sentence_list_json, sentence_dir, output_dir, video_clip_folder):
    video_name_list = load_txt(video_name_path)
    sentence_list = load_json(sentence_list_json)
    for vd in video_name_list:
        sentence_list_video = sentence_list[vd]
        sentence_list_video_unique = list(set(sentence_list_video)) # remove duplicates
        for s in sentence_list_video_unique: # unique sentence number
            text = load_txt(sentence_dir / f"{vd}_{s}.txt") # load the sentence from gpt_narr_text
            saliency_curve = get_saliency_curve(text, vd, video_clip_folder) # The anchor we used is the 
            print(f"shape of saliency_curve = {saliency_curve.shape}")
            save_path = Path(output_dir) / f"{vd}_{s}_saliency_curve.npy" # save saliency score
            np.save(save_path, saliency_curve)


# extract the saliency score from saliency curve

if __name__ == "__main__":
    pass