import torch
import subprocess
from PIL import Image
import pathlib 
from collections import Counter
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from scipy.stats import kendalltau, spearmanr
import sys
from pathlib import Path
import time
import open_clip
import utils
import torch.nn.functional as F
import argparse
import logging
import json
import glob
import copy
import torchvision
import os
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from utils import load_json, save_json, load_txt, save_txt
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, Normalize
import pandas as pd
import altair as alt
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm
import matplotlib.pyplot as plt


class LoadVideo(Dataset): # load one specific video
    def __init__(self, root, partition, transforms=None):   
        self.transforms = transforms 
        self.image_dir = root  # path of input directory
        self.image_files = sorted([f for f in self.image_dir.iterdir() if f.is_file()])
    def __getitem__(self, index):
        # Build the full file path for the target frame image using pathlib
        image_path = self.image_dir / self.image_files[index]
        
        # Load the image using PIL
        img = Image.open(image_path).convert('RGB')  # Convert to RGB to ensure consistency
        img_t = self.transforms(img)
        target_frame = index
        
        frame_rate = 1  # Frames per second
        timestamp = target_frame / frame_rate * 1000  # Convert to milliseconds
        
        return img_t, target_frame, timestamp
    
    def __len__(self):
        return len(self.image_files)



def confusion_mat(matched_frames_gen_np, matched_frames_intro_np, total_intro_frame, total_gen_frame):
    """
    confusion matrix
    """
    # print(f"shape of matched = {matched_frames_gen_np.shape}, {matched_frames_intro_np.shape}")
    common_elements = np.intersect1d(matched_frames_gen_np, matched_frames_intro_np)
    # common_elements = common_elements[common_elements != -1] # 0729: should change for tf remove -1 from the arry
    common_element_len = len(common_elements) # True positive

    return common_element_len

def is_dark(image, brightness_threshold=29.455): # should smaller than 29.455 are considered as dark
    
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.transpose(1, 2, 0) # h,w,c

    if image.ndim == 2:
        gray = image.astype(np.float32)
    else:

        gray = image.mean(axis=2).astype(np.float32) # average over channel dimension

    if gray.max() <= 1.0:
        gray = gray * 255.0

    avg_brightness = float(gray.mean())

    return avg_brightness < brightness_threshold



def find_num_clips_ratio_and_intervals(matched_frame_list):
    select_list = matched_frame_list.tolist()  # Convert to list
    # print(f"select_list = {select_list}")
    num_clips = 0
    intervals = []
    
    # Initialize variables to track the current clip
    current_clip = [select_list[0]]
    
    for i in range(1, len(select_list)):
        if select_list[i] == select_list[i - 1] + 1 or (select_list[i] == -1 and select_list[i - 1] == -1) or select_list[i] == select_list[i - 1]:
            # If the current frame is part of the same clip (either a sequence or consecutive -1s)
            current_clip.append(select_list[i])
        else:
            # End of the current clip, start a new one
            intervals.append(current_clip)
            num_clips += 1
            current_clip = [select_list[i]]
    
    # Append the last clip
    if current_clip:
        intervals.append(current_clip)
        num_clips += 1
    
    # Calculate the clip ratio
    num_clip_ratio = num_clips / len(select_list)  # Ratio: number of clips / total frames
    
    return num_clip_ratio

def find_indices(matrix):
    matrix = np.array(matrix)
    indices = []
    for i, row in enumerate(matrix):
        result = np.where(row == 1)[0]
        if len(result) == 0:
            indices.append(-1)  # Append -1 if no 1 is found in the row
        else:
            index = result[0]
            indices.append(index)
    return indices

def remove_all_black(frame_num_list, frame_folder):

    n_px = 224
    # when run clip feature during demo, roiginally use clip feature extraction
    preprocess = transforms.Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) # result image shape 224, 224

    # print(f"frame_num_list = {frame_num_list}")
    frame_num_list = frame_num_list.tolist() # convert to list
    main_dataset = LoadVideo(frame_folder, partition="main", transforms=preprocess)
    for frame_num in frame_num_list:
        frame, _, _ = main_dataset[frame_num] # should return numpy array
        if is_dark(frame):
            frame_num_list.remove(frame_num)
    return frame_num_list




def find_duplicates(input_list):
    # Calcualte Repetitiveness
    matched_frame = [item for item in input_list if item != -1]
    input_len = len(input_list)
    negative_one_count = input_len - len(matched_frame)

    total_elements = len(input_list)
    unique_elements_num = negative_one_count + len(set(matched_frame))
    res = 1 - (unique_elements_num / total_elements) # ratio of duplicates
    return res


def construct_csv(video_path, csv_save_path, search_dir, search_type, smoothed): # all current csv
 

    df = pd.DataFrame(columns=["Video_Name", "DemoFrameLen",  "OriIntroFrameLen", "MainFrameLen","OriMatchedFrameLen", "GenFramedMatchedLen","Ori_Gen_Intersect","LD_FIXED_ori","Ori_Intro_Clips", "Gen_Intro_Clips", "Ori Repeat Ratio", "Gen Repeat Ratio"])

    valid_video_names = load_txt(video_path)

    for video_name in valid_video_names:
        print(f"video name = {video_name}")
        main_frame_root_path = Path("/home/weihanx/videogpt/deepx_data7/frame_outputs_main_test") / video_name[0] / video_name
        main_dataset = LoadVideo(main_frame_root_path, partition="main", transforms=None)# vid_stride: how many frames will be escape, vid_stride=1:sequentially
        total_main_frame = len(main_dataset) # include all frames, 0908

        # use cos_l2_10 loss
        gt_dir = Path("/home/weihanx/videogpt/deepx_data6/actual_intro_cos_l2_20")
        gt_dir_path = gt_dir / f"{video_name}.npy"
        original_intro_matched = np.load(gt_dir_path)

        original_intro_matched = original_intro_matched.astype(int)
        ori_intro_clips = find_num_clips_ratio_and_intervals(original_intro_matched)

        assert original_intro_matched.size > 0, "The loaded original array is empty!"

        original_intro_matched = remove_all_black(original_intro_matched, main_frame_root_path) 

        ori_repeat = find_duplicates(original_intro_matched) # GT repetitiveness, consider those non-matched as separate new frames
        original_intro_matched = [item for item in original_intro_matched if item != -1]
        total_ori_intro_frame = len(original_intro_matched) # remove non-matched and black frame
        original_intro_matched_np = np.array(original_intro_matched) # for intesction



        if smoothed == True:
            load_path = search_dir / f"{video_name}_{search_type}_processed.npy" #this is returned by decode frame index
        else:
            load_path = search_dir / f"{video_name}_{search_type}.npy"

        gen_matched_np = np.load(load_path)

        gen_matched_np = gen_matched_np.astype(int)

        gen_intro_clips = find_num_clips_ratio_and_intervals(gen_matched_np) # scene change rate

        gen_matched_list = gen_matched_np.tolist()

        gen_repeat = find_duplicates(gen_matched_list) # repetitiveness

        total_gen_frame = len(gen_matched_list)


        original_intro_matched_len = len(original_intro_matched)
        gen_matched_len = len(gen_matched_list)

        common_element_len_ori = confusion_mat(gen_matched_np, original_intro_matched_np, total_ori_intro_frame, total_gen_frame)
        row = pd.DataFrame([[video_name, total_gen_frame,total_ori_intro_frame, total_main_frame, original_intro_matched_len, gen_matched_len, common_element_len_ori,LD_FIXED_ori, ori_intro_clips, gen_intro_clips, ori_repeat, gen_repeat]], columns=df.columns) # update 0824 only eval on those original introudction
        df = pd.concat([df, row], ignore_index=True)



    df.to_csv(csv_save_path, index=False)

def get_clip_score(video_name_path, input_array_dir, decode_method, body_contents_dir, sent_embed_dir, sent_list_json, output_dir, smoothed):
    sentence_list = load_json(sent_list_json)
    df = pd.DataFrame(columns=["Video_Name", "Clip_Score"])
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        frame_emb = []
        for i in range(10): # I divide them into 10 chunks for each video clip
            image_emb_path = body_contents_dir/ vd[0] / f"{vd}_clip_{i}" / f"{vd}_clip_{i}_body_clip.npy" # load all test video and do inference
            image_emb = np.load(image_emb_path)
            frame_emb.append(image_emb)
        combined_frame_emb = np.vstack(frame_emb)

        if smoothed == True:
            array_path = input_array_dir / f"{vd}_{decode_method}_processed.npy"
        if smoothed == False:
            array_path = input_array_dir / f"{vd}_{decode_method}.npy"
        if smoothed == "GT":
            array_path = input_array_dir / f"{vd}.npy"
        decoded_array = np.load(array_path)
        decoded_array = decoded_array.astype(int)

        sentence_list_video = sentence_list[vd]
        cosine_similarity_list = []
        lowest_range = min(len(decoded_array), len(sentence_list_video)) # only for baseline
        for i in range(lowest_range): # for one video
            sentence_id = sentence_list_video[i]
            frame_id = decoded_array[i]
            sentence_embedding = np.load(sent_embed_dir / f"{vd}_{sentence_id}.npy") # load sentence embedding
            if frame_id == -1:
                continue # do not calculate cosine similarity
            frame_embedding = combined_frame_emb[[frame_id]]
            # calculate cosine similarity
            cosine_similarity = F.cosine_similarity(torch.tensor(sentence_embedding), torch.tensor(frame_embedding), dim=1) # cosine similarity
            cosine_similarity_list.extend(cosine_similarity.flatten())
        # print(f"len of cosine_similarity_list = {len(cosine_similarity_list)}")
        row = pd.DataFrame([[vd, np.mean(cosine_similarity_list)]], columns=df.columns)
        df = pd.concat([df, row], ignore_index=True)
    average_dict = df.mean().to_dict()
    print(f"average_dict clip score = {average_dict}")
    df.to_csv(output_dir, index=False)
            # load sentence embedding

def get_clip_score_s(video_name_path, input_array_dir, decode_method, body_contents_dir, sent_embed_dir, sent_list_json, output_dir, smoothed):
    sentence_list = load_json(sent_list_json)
    df = pd.DataFrame(columns=["Video_Name", "Clip_Score"])
    # for fair comparison between two models: those are compare under B-32 
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        # print(f"video_name = {vd}")
        frame_emb = []
        for i in range(10): # I divide them into 10 chunks for each video clip
            image_emb_path = body_contents_dir/ vd[0] / f"{vd}_clip_{i}" / f"{vd}_clip_{i}_body_clip.npy" # load all test video and do inference
            image_emb = np.load(image_emb_path)
            frame_emb.append(image_emb)
        combined_frame_emb = np.vstack(frame_emb)
        # print(f"smoothed = {smoothed}")
        if smoothed == True:
            array_path = input_array_dir / f"{vd}_{decode_method}_processed.npy"
        if smoothed == False:
            array_path = input_array_dir / f"{vd}_{decode_method}.npy"
        if smoothed == "GT":
            array_path = input_array_dir / f"{vd}.npy"
        decoded_array = np.load(array_path)
        decoded_array = decoded_array.astype(int)
        # oad sentence list
        sentence_list_video = sentence_list[vd]
        cosine_similarity_list = []
        lowest_range = min(len(decoded_array), len(sentence_list_video)) # only for baseline
        for i in range(lowest_range): # for one video # baseline only
            sentence_id = sentence_list_video[i]
            frame_id = decoded_array[i]
            sentence_embedding = np.load(sent_embed_dir / f"{vd}_{sentence_id}.npy") # load sentence embedding
            if frame_id == -1:
                continue # do not calculate cosine similarity
            frame_embedding = combined_frame_emb[[frame_id]]
            cosine_similarity = F.cosine_similarity(torch.tensor(sentence_embedding), torch.tensor(frame_embedding), dim=1) # cosine similarity
            cosine_similarity_list.extend(cosine_similarity.flatten())
        cosine_similarity_list = np.clip(cosine_similarity_list, 0, None)
        cosine_similarity_list = 2.5 * cosine_similarity_list
        row = pd.DataFrame([[vd, np.mean(cosine_similarity_list)]], columns=df.columns)
        df = pd.concat([df, row], ignore_index=True)
    average_dict = df.mean().to_dict()
    print(f"average_dict clip score s, weight is 2.5 = {average_dict}")
    df.to_csv(output_dir, index=False)


def get_saliency_score(video_name_path, input_array_dir, decode_method, saliency_curve_dir, sent_list_json, output_dir, smoothed):

    df = pd.DataFrame(columns=["Video_Name", "Saliency_Score"])
    video_name_list = load_txt(video_name_path)
    sentence_indicator = load_json(sent_list_json)
    for vd in video_name_list:
        saliency_score = 0
        sentence_indicator_video = sentence_indicator[vd]
        if smoothed == True:
            array_path = input_array_dir / f"{vd}_{decode_method}_processed.npy"
        if smoothed == False:
            array_path = input_array_dir / f"{vd}_{decode_method}.npy"
        if smoothed == "GT":
            array_path = input_array_dir / f"{vd}.npy"
        decoded_array = np.load(array_path)
        decoded_array = decoded_array.astype(int)
        lowest_range = min(len(decoded_array), len(sentence_indicator_video)) # only for baseline, rounding cuz mismatch length
        for i in range(lowest_range): # for one video
            sentence_id = sentence_indicator_video[i]
            frame_id = decoded_array[i] # each sentences has one frame in decoded array

            saliency_path = saliency_curve_dir / f"{vd}_{sentence_id}_saliency_curve.npy"
            saliency_score_curve = np.load(saliency_path)
            saliency_score += saliency_score_curve[frame_id] # accumulate saliency score
        saliency_score = saliency_score / len(sentence_indicator_video) # average saliency score
        row = pd.DataFrame([[vd, saliency_score]], columns=df.columns)
        df = pd.concat([df, row], ignore_index=True)
    average_dict = df.mean().to_dict()
    print(f"average_dict saliency score = {average_dict}")
    df.to_csv(output_dir, index=False)

def generate_sentence_json_a2summ(video_name_path, output_dir):
    """
    "rslUfl6OsV4": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "2", "5", "5", "5", "5", "5", "5", "5", "5", "5", "8", "8", "8", "9", "9", "9", "9", "9", "9", "9", "9", "10", "10", "10", "10", "10", "10", "10", "10", "10", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "11", "16", "16", "16", "16", "16", "16", "16", "16", "16"]
    """
    video_name_list = load_txt(video_name_path)
    sentence_dict = {}

    for vd in video_name_list:
        sentence_dict[vd] = []
        # 8 * [0] + 8 * [1]
        for i in range(10):
            # 8 times
            sentence_dict[vd].extend([str(i)] * 8)
    output_path = output_dir / "a2summ_gpt_sentence_list.json"
    save_json(output_path, sentence_dict)


if __name__ == "__main__":
    # video_path, csv_save_path, search_type

    # # # ######## gpt, beam search #########
    # print(f"Evaluating --- gpt_beamsearchwindow_0918_point3")
    # decode_type = "beam_search"
    # folder_name = "gpt_beamsearchwindow_0918_point3"
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    # sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path)

    # print(f"Evaluating --- gpt_beamsearchwindow_0918_point5")
    # decode_type = "beam_search"
    # folder_name = "gpt_beamsearchwindow_0918_point5"
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    # sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path)

    # print(f"Evaluating --- gpt_beamsearchwindow_0919_point5_dp")
    # decode_type = "beam_search"
    # folder_name = "gpt_beamsearchwindow_0919_point5_dp"
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    # sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path)

    # print(f"Evaluating --- gpt_beamsearchwindow_0918_point7")
    # decode_type = "beam_search"
    # folder_name = "gpt_beamsearchwindow_0918_point7"
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    # sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path)

    # print(f"Evaluating --- gpt_greedy_sentence_0919_dp")
    # decode_type = "greedy_sentence"
    # folder_name = "gpt_greedy_sentence_0919_dp"
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    # sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path)

    # print(f"Evaluating --- gpt_beamsearchwindow_0918_1")
    # decode_type = "beam_search"
    # folder_name = "gpt_beamsearchwindow_0918_1"
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    # sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path)


    # decode_type = "beam_search"
    # gpt = False  
    # folder_name = "ori_beamsearchwindow_0926_1"
    # print(f"Evaluating --- {folder_name}")
    # smoothed = True # need "processed" in the file name
    # if gpt == False:
    #     sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
    #     sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_list.json")
    #     saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/demo/saliency_curve_0925")
    # if gpt == True:
    #     sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
    #     sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
    #     saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve_0925")
    # video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # csv_save_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type, smoothed)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore.csv")
    # get_clip_score_s(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path, smoothed)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/clipscore_s.csv")
    # get_clip_score(video_path, search_dir, decode_type, input_body_dir, sent_embed_dir,sent_list_json, output_csv_path, smoothed)
    # output_csv_path = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/{folder_name}/saliency.csv")
    # get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_csv_path, smoothed)

# Evaluate on zeroshot
# # /home/weihanx/videogpt/deepx_data6/transformer_prior/test_sentence_zeroshot.json
#     video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
#     input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
#     csv_save_path = Path("/home/weihanx/videogpt/deepx_data6/transformer_prior/zeroshot_f1.csv")
#     sent_list_json = Path("/home/weihanx/videogpt/deepx_data6/transformer_prior/test_sentence_zeroshot.json")
#     sent_embed_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip_frame_768_emb")
#     search_dir = Path("/home/weihanx/videogpt/ICLR2025/zeroshot_dp")
#     construct_csv(video_path , csv_save_path, search_dir, decode_type, smoothed)
# # Evaluate on CSTA
#     smoothed = False
#     decode_type = "csta"
#     video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
#     input_body_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
#     csv_save_path = Path(f"/home/weihanx/videogpt/deepx_data6/baseline_models/CSTA/f1.csv")
#     search_dir = Path(f"/home/weihanx/videogpt/deepx_data6/baseline_models/CSTA/csta_baseline")
#     construct_csv(video_path , csv_save_path, search_dir, decode_type, smoothed)

    # CSTA zero shot
    # smoothed = False
    # decode_type = "csta"
    # video_path = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_name.txt")
    # csv_save_path = Path(f"/home/weihanx/videogpt/deepx_data6/baseline_models/CSTA/f1.csv")
    # search_dir = Path(f"/home/weihanx/videogpt/deepx_data6/baseline_models/CSTA/csta_baseline_zeroshot")
    # construct_csv(video_path , csv_save_path, search_dir, decode_type, smoothed)   
    # asumm zero shot
    smoothed = False
    decode_type = "a2summ"
    video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    csv_save_path = Path(f"/home/weihanx/videogpt/deepx_data6/testset_a2summ/f1.csv")
    search_dir = Path(f"/home/weihanx/videogpt/deepx_data6/testset_a2summ/decoded_array")
    body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb_512")
    sent_embed_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/sentence_text_embed")
    saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/saliency_curve")
    construct_csv(video_path , csv_save_path, search_dir, decode_type, smoothed)  
    sent_list_json = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/a2summ_gpt_sentence_list.json")
    get_clip_score(video_path, search_dir, decode_type, body_contents_dir, sent_embed_dir, sent_list_json, output_dir=None, smoothed=False)
    get_clip_score_s(video_path, search_dir, decode_type, body_contents_dir, sent_embed_dir, sent_list_json, output_dir=None, smoothed=False)
    get_saliency_score(video_path, search_dir, decode_type, saliency_curve_dir, sent_list_json, output_dir=None, smoothed=False)
    # ## For Ground Truth
    # decode_method = ""
    # input_array_dir = Path("/home/weihanx/videogpt/deepx_data6/actual_intro_cos_l2_20")
    # body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # sent_embed_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")

    # saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/demo/saliency_curve_0925")

    # get_clip_score(video_name_path, input_array_dir, decode_method, body_contents_dir, sent_embed_dir, sent_list_json, output_dir=None, smoothed="GT")
    # get_clip_score_s(video_name_path, input_array_dir, decode_method, body_contents_dir, sent_embed_dir, sent_list_json, output_dir=None, smoothed="GT")
    # get_saliency_score(video_name_path, input_array_dir, decode_method, saliency_curve_dir, sent_list_json, output_dir=None, smoothed="GT")
