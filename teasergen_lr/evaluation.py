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
    def __init__(self, root, transforms=None):   
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




def levenshtein_distance(list1, list2):
    # Create a matrix to store distances
    # need to remove all negative 1
    # print(f"unfiltered = {list1}, {list2}")
    list1 = [item for item in list1 if item != '-1']
    list2 = [item for item in list2 if item != '-1']
    # print(f"filtered = {list1}, {list2}")
    rows = len(list1) + 1
    cols = len(list2) + 1
    distance_matrix = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # Initialize the first row and column
    for i in range(1, rows):
        distance_matrix[i][0] = i
    for j in range(1, cols):
        distance_matrix[0][j] = j
    
    # Populate the matrix
    for i in range(1, rows):
        for j in range(1, cols):
            if list1[i-1] == list2[j-1]:
                cost = 0
            else:
                cost = 1
            distance_matrix[i][j] = min(distance_matrix[i-1][j] + 1,      # Deletion
                                        distance_matrix[i][j-1] + 1,      # Insertion
                                        distance_matrix[i-1][j-1] + cost) # Substitution
    
    # The last cell of the matrix contains the Levenshtein distance
    return distance_matrix[-1][-1]

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
    # Check if image is a torch tensor and convert to numpy if needed
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    grayscale_image = np.mean(image, axis=2)
    
    # Calculate the average brightness
    avg_brightness = np.mean(grayscale_image)
    
    return avg_brightness < brightness_threshold



def find_num_clips_ratio_and_intervals(matched_frame_list):
    select_list = matched_frame_list # Convert to list
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

def remove_all_black(frame_num_arr, frame_folder):
    """
    frame_num_arr array
    """
    n_px = 224
    # when run clip feature during demo, roiginally use clip feature extraction
    preprocess = transforms.Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]) # result image shape 224, 224


    frame_num_list = frame_num_arr.tolist() # convert to list
    main_dataset = LoadVideo(frame_folder, transforms=preprocess)
    for frame_num in frame_num_list:
        frame, _, _ = main_dataset[frame_num] # should return numpy array
        if is_dark(frame):
            frame_num_list.remove(frame_num)

    return frame_num_list

def find_topk_values(topk, matrix): # the matrix is from left to right, the value is getting larger
    matrix = np.array(matrix)
    values = []
    
    for row in matrix:
        # Get the values from the first `topk` columns directly
        topk_values = row[:topk]
        values.append(topk_values.tolist())
    # [[283], [284], [285], [286], [287], [288], [293], [294], [295], [1443], [1444], [1445], [1467], [1468], [1469], [1470], [648], [648], [650], [650], [732], [733], [733], [929], [930], [931], [134], [135], [136], [137], [746], [745], [740], [1163], [1164], [1165], [1186], [1187], [1188]]
    flat_list = [item for sublist in values for item in sublist] # convert to a single line
    return flat_list


def find_duplicates(input_list):
    # input is a lits
    # find unique elements
    # only gt has -1
    unique_elements = [item for item in input_list if item != -1]
    input_len = len(input_list)
    negative_one_count = input_len - len(unique_elements)
    # unique_elements = set(unique_elements) # -1 is alsp a unique value for GT
    total_elements = len(input_list)
    unique_elements_num = negative_one_count + len(set(unique_elements))
    # res = 1 - (len(unique_elements) / total_elements) # ratio of duplicates
    res = 1 - (unique_elements_num / total_elements) # ratio of duplicates
    return res

def calculate_correlations(list1, list2):
    # Calculate Kendall's Tau
    print(f"ori = {len(list1)}, gen = {len(list2)}")
    assert len(list1) == len(list2)
    # masked out those -1 in list1
    valid_indices = [i for i, val in enumerate(list1) if val != -1] # mask out those -1 in list 1
    filtered_list1 = [list1[i] for i in valid_indices]
    filtered_list2 = [list2[i] for i in valid_indices]
    assert len(filtered_list1, filtered_list2)
    kendall_tau, _ = kendalltau(filtered_list1, filtered_list2)
    
    # Calculate Spearman's Rho
    spearman_rho, _ = spearmanr(filtered_list1, filtered_list2)
    
    return kendall_tau, spearman_rho


def construct_csv(video_path, ground_truth_dir, search_dir, search_type, smoothed, body_frame_dir, csv_save_path):
    # Load list of video names
    valid_video_names = load_txt(video_path)
    failed_files = []
    df = pd.DataFrame(columns=["Video_Name", "GenFrameLen", "OriIntroFrameLen", "Common Element Len", "Ori_Intro_Clips", "Gen_Intro_Clips", "Ori Repeat Ratio", "Gen Repeat Ratio"])

    for video_name in valid_video_names:
        # Load ground truth and convert to int
        gt_path = Path(ground_truth_dir) / f"{video_name}.npy"
        original_intro_matched_arr = np.load(gt_path).astype(int)

        # Remove black frames, keeping only valid frames
        main_frame_root_path = Path(body_frame_dir) / video_name[0] / video_name
        original_intro_matched_list = remove_all_black(original_intro_matched_arr, main_frame_root_path).tolist()
        original_intro_matched_no_black = [item for item in original_intro_matched_list if item != -1]

        # Calculate statistics for original intro
        ori_intro_clips, ori_repeat = find_num_clips_ratio_and_intervals(original_intro_matched_no_black), find_duplicates(original_intro_matched_no_black)

        # Load generated matched frames
        processed_suffix = '_processed' if smoothed else ''
        load_path = Path(search_dir) / f"{video_name}_{search_type}{processed_suffix}.npy"
        gen_matched_arr = np.load(load_path).astype(int)
        if gen_matched_arr.size == 0:
            print(f"Fail video_name = {video_name}")
            failed_files.append(video_name)
            continue

        gen_intro_clips, gen_repeat = find_num_clips_ratio_and_intervals(gen_matched_arr.tolist()), find_duplicates(gen_matched_arr.tolist())

        # Calculate common elements
        common_element_len_ori = confusion_mat(gen_matched_arr, np.array(original_intro_matched_no_black), len(original_intro_matched_no_black), len(gen_matched_arr.tolist()))

        # Collect data for the current video
        row = pd.DataFrame([[
            video_name, len(gen_matched_arr), len(original_intro_matched_no_black), common_element_len_ori,
            ori_intro_clips, gen_intro_clips, ori_repeat, gen_repeat
        ]], columns=df.columns)
        df = pd.concat([df, row], ignore_index=True)

    # Calculate averages and output to CSV
    average_dict = df.mean().to_dict()
    print(f"Repetitiveness = {average_dict['Gen Repeat Ratio']}")
    print(f"Gen intro clips = {average_dict['Gen_Intro_Clips']}")
    print(f"Average metrics = {average_dict}")
    df.to_csv(csv_save_path, index=False)


def get_clip_score_s(video_name_path, input_array_dir, decode_method, body_contents_emb, sent_embed_dir, sent_list_json, output_dir, smoothed):
    sentence_list = load_json(sent_list_json)
    df = pd.DataFrame(columns=["Video_Name", "Clip_Score"])
    # for fair comparison between two models: those are compare under B-32 
    video_name_list = load_txt(video_name_path)
    for vd in video_name_list:
        # print(f"video_name = {vd}")
        combined_frame_emb = np.load(body_contents_emb / f"{vd}_clip.npy")

        if smoothed == True:
            array_path = input_array_dir / f"{vd}_{decode_method}_processed.npy"
        if smoothed == False:
            array_path = input_array_dir / f"{vd}_{decode_method}.npy"
        if smoothed == "GT":
            array_path = input_array_dir / f"{vd}.npy"
        decoded_array = np.load(array_path)
        decoded_array = decoded_array.astype(int)

        sentence_list_video = sentence_list[vd] # belong to which sentence

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
    search_type = "beam_search"
    narration_type = "gpt"
    smoothed = True
    ### Annotation ###
    video_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    ground_truth_dir = Path("/home/weihanx/videogpt/deepx_data6/actual_intro_cos_l2_20")
    folder_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/decode_result")
    body_frame_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/test_embedding/clip_frame_768_emb_main")
    sent_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test/test_gpt_sentence_list.json")
    ##### Output ####
    search_dir = folder_dir / narration_type / search_type
    csv_save_path = search_dir / f"smooth_{smoothed}_stats.csv"
    construct_csv(video_path, ground_truth_dir, search_dir, search_type, smoothed, body_frame_dir, csv_save_path)

    saliency_curve_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/saliency_curve")
    output_csv_path = search_dir / f"smooth_{smoothed}_saliency.csv"
    get_saliency_score(video_path, search_dir, search_type, saliency_curve_dir, sent_list_json, output_csv_path)

    sent_embed_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/test_embedding/clip_sentence_emb")
    output_csv_path = search_dir / f"smooth_{smoothed}_clipscore_s.csv"
    get_clip_score_s(video_path, search_dir, search_type, sent_embed_dir, sent_list_json, output_csv_path)
