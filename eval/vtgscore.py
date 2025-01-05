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

    video_clip_folder = Path("/home/weihanx/videogpt/deepx_data6/demo/video_clip")
    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_list.json")
    # sentence_dir = Path("/home/weihanx/videogpt/workspace/ori_narr_text")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/demo/saliency_curve_0925")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # get_array_saliency_score(video_name_path, sentence_list_json, sentence_dir, output_dir, video_clip_folder)
    
    # for zero shot
    # video_clip_folder = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip")
    video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    sentence_list_json = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/a2summ_gpt_sentence_list.json")
    
    sentence_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/sentence")
    output_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/saliency_curve")
    output_dir.mkdir(parents=True, exist_ok=True)
    get_array_saliency_score(video_name_path, sentence_list_json, sentence_dir, output_dir, video_clip_folder)
    
    # video_clip_folder = Path("/home/weihanx/videogpt/deepx_data6/demo/video_clip")
    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # sentence_dir = Path("/home/weihanx/videogpt/workspace/gpt_narr_text")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve_0925")
    # output_dir.mkdir(parents=True, exist_ok=True)
    # get_array_saliency_score(video_name_path, sentence_list_json, sentence_dir, output_dir, video_clip_folder)

    # video_name = "j1Pb8YU8wQo"
    # text = "The Mediterranean was once a major crossroads at the heart of the ancient world."
    # body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # frame_emb = []
    # # print(f"body_contents_dir: {body_contents_dir}, match_video_name: {match_video_name}")
    # for i in range(10): # I divide them into 10 chunks for each video clip
    #     image_emb_path = body_contents_dir/ video_name[0] / f"{video_name}_clip_{i}" / f"{video_name}_clip_{i}_body_clip.npy" # load all test video and do inference
    #     image_emb = np.load(image_emb_path)
    #     frame_emb.append(image_emb)
    # combined_frame_emb = np.vstack(frame_emb)

    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # image_embedding_folder = Path("/home/weihanx/videogpt/workspace/clip_image_emb_full")
    # sentence_embedding_folder = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_0.npy"
    # image_embedding_np = np.load(image_embedding)
    # sentence_embedding_np = np.load(sentence_embedding)
    # selected_embedding = image_embedding_np[2:7]
    # from sklearn.metrics.pairwise import cosine_similarity

    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_0.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # top3_indices = np.argsort(sentence_similarity)[-3:][::-1]  # Get top 3 indices in descending order
    # # top3_similarities = sentence_similarity[top3_indices]  # Corresponding similarity values

    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()} selected frame simialrity = {sentence_similarity.squeeze()[44:47]}, {sentence_similarity.squeeze()[1275:1278]}")



    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_1.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # top3_indices = np.argsort(sentence_similarity)[-3:][::-1]  # Get top 3 indices in descending order
    # # top3_similarities = sentence_similarity[top3_indices]  # Corresponding similarity values

    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}, {sentence_similarity.squeeze()[574:578]}, {sentence_similarity.squeeze()[1172:1175]}, {sentence_similarity.squeeze()[1273:1275]}")


    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_4.npy"

    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # top3_indices = np.argsort(sentence_similarity)[-3:][::-1]  # Get top 3 indices in descending order
    # # top3_similarities = sentence_similarity[top3_indices]  # Corresponding similarity values

    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}, {sentence_similarity.squeeze()[1423:1425]}, {sentence_similarity.squeeze()[1428:1431]}, {sentence_similarity.squeeze()[1454:1456]}")

    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_6.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # top3_indices = np.argsort(sentence_similarity)[-3:][::-1]  # Get top 3 indices in descending order
    # # top3_similarities = sentence_similarity[top3_indices]  # Corresponding similarity values

    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}, {sentence_similarity.squeeze()[942:946]}, {sentence_similarity.squeeze()[65:67]}, {sentence_similarity.squeeze()[191:193]}")


    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_7.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # top3_indices = np.argsort(sentence_similarity)[-3:][::-1]  # Get top 3 indices in descending order
    # # top3_similarities = sentence_similarity[top3_indices]  # Corresponding similarity values

    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}, {sentence_similarity.squeeze()[604:608]}")

    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy" # this is the short intro clip. Their cliscore is around 0.3
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_8.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}, {sentence_similarity.squeeze()[1454:1458]}")

    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_10.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}")

    # image_embedding = image_embedding_folder / f"{video_name}_clip.npy"
    # sentence_embedding = sentence_embedding_folder / f"{video_name}_11.npy"
    # sentence_similarity = cosine_similarity(sentence_embedding_np, combined_frame_emb)
    # # print(f"simiarlity = {sentence_similarity}")
    # print(f"selected_embedding = {sentence_similarity.shape}, sentence_embedding max = {sentence_similarity.max()}")

    # text = "Travels in Israel Mediterranean journey"

    # print(curve.shape)
    # # text = "The Mediterranean was once a major crossroads at the heart of the ancient world"
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[44:47].mean()}, {curve[1275:1278].mean()}, random = {curve[557:563].mean()}")
    # # text = "Today it has become a barrier separating Europe from Africa.  Is there anything left of a past once shared and what do today's distinct cultures  have in common?"
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[574:578].mean()}, {curve[1172:1175].mean()}, {curve[1273:1275].mean()},random{curve[159:170].mean()}")
    # # text = "Journalist Zinaq El-Masrach and Jafar Abdul-Karim traveled the coasts of the Mediterranean  in search of answers."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[1423:1425].mean()}, {curve[1428:1431].mean()}, {curve[1454:1456].mean()} random, {curve[1202:1210].mean()}")
    # # text = " Do you see yourself as a Tunisian Jew?"
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[942:946].mean()}, random, {curve[738:741].mean()}")
    # # text = " Yes, with all the rights and responsibilities."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score =  {curve[65:67].mean()}, {curve[191:193].mean()}, random, {curve[102:106].mean()}")
    # # text = "  How can you feel the food for these animals?  God helps us."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[604:608].mean()} random, {curve[45:50].mean()}")
    # # text = " Join us to get to know the people and their dreams."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[1454:1458].mean()}random, {curve[1350:1354].mean()}")
    # # text = "A Mediterranean journey."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[664:666].mean()}, random, {curve[779:781].mean()}")
    # # load embedding to get cosine simiarlity

    # # text = "The Mediterranean was once a major crossroads at the heart of the ancient world"
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[664:666].mean()}, {curve[1169:1171].mean()}, {curve[1173:1175].mean()}random = {curve[557:563].mean()}")
    # # text = "Today it has become a barrier separating Europe from Africa.  Is there anything left of a past once shared and what do today's distinct cultures  have in common?"
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[289:291].mean()}, {curve[746:748].mean()}, {curve[1101:1103].mean()},{curve[1171:1173].mean()},random{curve[159:170].mean()}")
    # # text = "Journalist Zinaq El-Masrach and Jafar Abdul-Karim traveled the coasts of the Mediterranean  in search of answers."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[1050:1053].mean()}, {curve[1058:1062].mean()} random, {curve[1202:1210].mean()}")
    # # text = " Do you see yourself as a Tunisian Jew?"
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[470:472].mean()}, {curve[664:666].mean()} random, {curve[738:741].mean()}")
    # # text = " Yes, with all the rights and responsibilities."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score =  {curve[729:732].mean()}, random, {curve[102:106].mean()}")
    # # text = "  How can you feel the food for these animals?  God helps us."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[737:739].mean()} {curve[741:744].mean()} random, {curve[45:50].mean()}")
    # # text = " Join us to get to know the people and their dreams."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[306:308].mean()} {curve[359:361].mean()} random, {curve[1350:1354].mean()}")
    # # text = "A Mediterranean journey."
    # curve = get_saliency_curve(text, video_name, video_clip_folder)
    # print(f"salinecy score = {curve[729:732].mean()}, random, {curve[779:781].mean()}")
    # # load embedding to get cosine simiarlity
    # text = "sal_curve_ The Mediterranean was once a major crossroads at the heart of the ancient world."
    # video_name = "j1Pb8YU8wQo"
    # video_clip_folder = Path("/home/weihanx/videogpt/deepx_data6/demo/video_clip")
    # get_saliency_curve(text, video_name, video_clip_folder)