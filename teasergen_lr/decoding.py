# feature extraction for all subclips in test set
from pathlib import Path
import numpy as np
import torch
import bisect
from transformers import CLIPProcessor, CLIPModel, AutoProcessor
from utils import load_txt, save_txt, load_json, save_json
import json
import sys
from pathlib import Path
import torch
from utils import save_json, load_json, save_txt, load_txt
import os
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop, ToTensor
from transformers import RobertaModel, RobertaTokenizer
from pathlib import Path


def nearest_neighbour(topk_indices, input_image_emb, match_video_name, body_contents_dir):
    # I have 10 video_clisp for each video_ame
    print(f"I keep {topk_indices} for this output embedding")
    combined_frame_emb = np.load(body_contents_dir / f"{match_video_name}_clip.npy")
    total_frame_in_video = combined_frame_emb.shape[0]

    input_image_emb = input_image_emb.detach().cpu().numpy()
    print(f"input_image_emb shape: {input_image_emb.shape}")
    cosine_similarities_list = cosine_similarity(input_image_emb, combined_frame_emb) # (1, 2048) * (132, 2048) -> (1, 132)
    print(f"cosine_similarities_list shape: {cosine_similarities_list.shape}") # should be total number of frames
    k_nearest_neighbour = np.argsort(cosine_similarities_list, axis=1)[:, -topk_indices:]  # Get the last k indices (highest similarities)
    # Reverse the order to get indices in descending order of similarity
    k_nearest_neighbour = k_nearest_neighbour[:, ::-1]
    knn_emb = combined_frame_emb[k_nearest_neighbour]
    knn_emb = knn_emb.squeeze(0) #(1,512)
    # print(f"knn_emb: {knn_emb.shape} and k_nearest_neighbour: {k_nearest_neighbour.shape}") #(1,512) and (1, #topk)
    return k_nearest_neighbour[0], total_frame_in_video # from most similar to least similar # return the index of the frame, return: [318, 200, 213]

def find_positions_descending(grid):
    # Convert the grid to a NumPy array for easier manipulation
    grid_array = np.array(grid)
    
    # Flatten the array and get the indices sorted by value in descending order
    sorted_indices = np.argsort(grid_array, axis=None)[::-1]
    
    # Convert the flattened indices back to 2D row, column indices
    rows, columns = np.unravel_index(sorted_indices, grid_array.shape)
    
    # Create a list of (row, column) tuples
    sorted_positions = []
    sorted_next_positions = []
    for r, c in zip(rows, columns):
        sorted_next_positions.append(c)
        sorted_positions.append((r,c))
    return sorted_positions, sorted_next_positions


def beam_search_index(beam_size_former, last_top_k_indices, current_embedding, beam_size_later, lamada, match_video_name, body_contents_dir, diversify=True):
    """
    Goal: I want to have the most salient frames and least repetitive frames
    return: the decoded index for embedding to compare
    current_embedding: 0901: already moved out the batch dimension
    """
    # last_selected_embedding: (3, 512) 
    combined_frame_emb = np.load(body_contents_dir / f"{match_video_name}_clip.npy")
    current_embedding = current_embedding.detach().cpu().numpy()
    cosine_similarities_list = cosine_similarity(current_embedding, combined_frame_emb) # (1, 2048) * (132, 2048) -> (1, 132) # select top 10 
    # print(f"current_embedding: {current_embedding.shape}, combined_frame_emb: {combined_frame_emb.shape}")
    # print(f"cosine_similarities_list: {cosine_similarities_list.shape}") # 1, 686
    top_k_indices = np.argsort(cosine_similarities_list, axis=1)[:, -beam_size_later:]  # Get the last k indices (highest similarities) from small to large
    # print(f"top_k_indices: {top_k_indices}")
    top_k_indices = top_k_indices[:, ::-1] # reverse the order, descending order
    print(f"top_k_indices: {top_k_indices.shape}") # (1, 10)
    top_k_similarity = cosine_similarities_list[0][top_k_indices]  # Get the similarity scores for the top k indices
    selected_frames = combined_frame_emb[top_k_indices] # (1, 3, 512) # top 10
    selected_frames = selected_frames.squeeze(0)
    print(f"selected_frames: {selected_frames.shape}")
    # least similar to last selected embedding
    # need to get cross similarity between last selected embedding and selected frames
    double_beam_search = []
    # score function ranking
    for i in last_top_k_indices: 
        last_selected_i = combined_frame_emb[[i]] # 1,512

        diveristy = cosine_similarity(last_selected_i, selected_frames).squeeze(0) # (1, 512) * (10, 512) --> (10, )

        closeness = top_k_similarity.squeeze(0)  # (10, )
        # print(f"closeness: {closeness.shape}")
        if diversify == False:
            score = closeness 
            # + lamada * diveristy # selected last step such that it is more related to my current step, they inheritbly really similar
        else:
            score = - lamada * diveristy + closeness # I want to maximize the score, bigger diversity and smaller cosine simlairty
        # print(f"score: {score.shape}") # 10,
        score = score[np.newaxis,:]
        # print(f"score: {score.shape}") # 1, 10
        double_beam_search.append(score) # (1, 10) 
    double_beam_search = np.vstack(double_beam_search) # (3, 10)

    sorted_positions, sorted_next_position = find_positions_descending(double_beam_search) # this is the position in grid

    last_position = last_top_k_indices[sorted_positions[0][0]]
    # print(f"last_position: {last_position} should be in last top k dimension")
    current_top_beam_former_index = top_k_indices[0][sorted_next_position[:beam_size_former]] # get the top k indices, in descending order
    # print(f"least_embedding: {least_embedding.shape}") # (1, 512)
    # print(f"current_top_beam_former_index: {current_top_beam_former_index}")
    return last_position, current_top_beam_former_index # top former score


def beam_search_history(decode_index, beam_size_former, history_dict, current_embedding, beam_size_later, lamada, match_video_name, body_contents_dir,sentence_list):
    """
    Goal: I want to have the most salient frames and least repetitive frames
    return: the decoded index for embedding to compare
    current_embedding: 0901: already moved out the batch dimension
    sentence_list: denote whether they are in the same sentence
    """
    # last_selected_embedding: (3, 512) 
    combined_frame_emb = np.load(body_contents_dir / f"{match_video_name}_clip.npy")
    total_frame_in_video = combined_frame_emb.shape[0]
    # most similar to current embedding
    current_embedding = current_embedding.detach().cpu().numpy()
    cosine_similarities_list = cosine_similarity(current_embedding, combined_frame_emb) # (1, 2048) * (132, 2048) -> (1, 132) # select top 10 

    top_k_indices = np.argsort(cosine_similarities_list, axis=1)[:, -beam_size_later:]  # Get the last k indices (highest similarities) from small to large
    # print(f"top_k_indices: {top_k_indices}")
    top_k_indices = top_k_indices[:, ::-1] # reverse the order, descending order
    # print(f"top_k_indices: {top_k_indices}") # (1, 10)
    top_k_similarity = cosine_similarities_list[0][top_k_indices]  # Get the similarity scores for the top k indices
    selected_frames = combined_frame_emb[top_k_indices] # (1, 3, 512) # top 10
    selected_frames = selected_frames.squeeze(0)

    double_beam_search = []
    # score function ranking
    for key, value in history_dict.items():

        score = top_k_similarity.squeeze(0) # saliency result
        # print(f"score = {score.shape}")
        cur_sent_num = sentence_list[decode_index]
        for i in range(decode_index-1, -1, -1):
            sent_num = sentence_list[i] # during inference, I only know different sentences should different scene. I don't know within sentences. Within sentence should be handled by the model

            # if value[i] == total_frame_in_video:
            prev_selected_frame = combined_frame_emb[[value[i]]]
            if sent_num != cur_sent_num:
                decrement = lamada * cosine_similarity(prev_selected_frame, selected_frames).squeeze(0)
                # print(f"decrment shape = {decrement.shape}")
                score = score - decrement
        score = score[np.newaxis,:] # should be 10
        # print(f"score: {score.shape}") # 1, 10
        double_beam_search.append(score) # (1, 10) 
    
    double_beam_search = np.vstack(double_beam_search) # should be 30
    # print(f"double = {double_beam_search.shape}")
    sorted_positions, sorted_next_position = find_positions_descending(double_beam_search) # this is the position in grid
    # rank on score function
    highest_beam = sorted_positions[:beam_size_former] # because default in descending order, but this position is not in body contents dimension
    # r,c should be the position in topk
    new_history_dict = {}
    for idx, (r,c) in enumerate(highest_beam):
        # print(f"idx = {idx}, r = {r} , c = {c}, history_dict = {len(history_dict[r])}")
        new_history_dict[idx] = history_dict[r].copy()
        # print(f"len() = {len(new_history_dict[idx])}")
        # print(f"c = {top_k_indices[0][c]}")
        new_history_dict[idx].append(top_k_indices[0][c])
        if top_k_indices[0][c] >= total_frame_in_video:
            print(f"Warning, found some frames indices out of boundary")
            sys.exit()

    return new_history_dict, total_frame_in_video
    # the three highest one will be retured



def decode_greedy_sentence(topk_indices, output_embedding, mask, match_video_name, body_contents_dir, output_dir, sentence_list_json,batch_size=1):

    total_frames_to_decode = output_embedding.shape[1]
    batch_size = output_embedding.shape[0]
    decoded_frame_index = []

    for batch_idx in range(batch_size):
        current_video_name = match_video_name[batch_idx]
        current_batch_decoded_indices = []
        for i in range(total_frames_to_decode):

            # Process the ith frame in the current batch
            nn_idx, total_frame_in_video = nearest_neighbour(
                topk_indices, 
                output_embedding[batch_idx, i].unsqueeze(0),  # (1,512), seq dimenions
                current_video_name, 
                body_contents_dir
            )
                # print(f"Batch {batch_idx}, i = {i}, selected_frame: {nn_idx}, match_video_name = {match_video_name}")
            current_batch_decoded_indices.append(nn_idx[0])  # Append the most salient one in the body contents

        decoded_frame_index = np.array(current_batch_decoded_indices)
        # deal with repetitiveness
        output_path = output_dir / f"{current_video_name}_greedy.npy"
        
        np.save(output_path, decoded_frame_index)

        result = consecutive_to_interval(decoded_frame_index, current_video_name, total_frame_in_video)

        # greedy_sentence_demo_time_interval(decoded_frame_index, current_video_name, time_duration_folder, sentence_list_json,demo_duration_output_dir)
        output_path = output_dir / f"{current_video_name}_greedy_processed.npy"
        print(f"video_name: {current_video_name}, decoded_frame_index: {decoded_frame_index}, save = {result}")

        # print(f"video_name: {current_video_name}, decoded_frame_index: {decoded_frame_index}")
        np.save(output_path, result)


def greedy_sentence_demo_time_interval(decode_frame_list, video_name, time_duration_dir, sentence_list_json, demo_duration_output_dir):
    # time_duration_dir: use random time duration
    time_duration_dict = load_json(time_duration_dir/video_name[0]/ video_name/"time_interval_corrected.json")
    # sentence id
    sentence_list_dict = load_json(sentence_list_json)
    sentence_list = sentence_list_dict[video_name] # the sentence list for this particular video
    demo_time_interval_dict = {}
    # also need sentence_list
    current_sentence_value = -1
    print(f"sentence_list: {sentence_list}, decode_frame_list: {decode_frame_list}")
    assert len(sentence_list) == len(decode_frame_list), "sentence_list and decode_frame_list should have the same length"
    for s_idx in range(len(sentence_list)): # [1,1,1,1,2,2,2,3,3,3]
        if sentence_list[s_idx] != current_sentence_value: # the first index
            print(f"sentence_list[s_idx]: {sentence_list[s_idx]}")
            expect_duration_interval = time_duration_dict[sentence_list[s_idx]][0]
            expect_duration = expect_duration_interval[1] - expect_duration_interval[0] # start - end: inclusive
            demo_time_interval_dict[sentence_list[s_idx]] = [[float(decode_frame_list[s_idx]), round(float(decode_frame_list[s_idx] + expect_duration), 4)]]# inclusive
            current_sentence_value = sentence_list[s_idx] # if add this line, then it is the start poisition frames, if remove, then use the last frame
    save_json(demo_duration_output_dir / f"{video_name}_greedy_sentence_time_interval.json", demo_time_interval_dict)

def decode_beam_search_history(topk_indices, output_embedding, mask, match_video_name, body_contents_dir, beam_size_former, beam_size_later, output_dir, lamda, sentence_list_json, batch_size=1):
    # beam size former: the first bean size
    # beam size later: the second beam size
    # lamda is used to weight diversity and closeness
    total_frames_to_decode = output_embedding.shape[1]
    batch_size = output_embedding.shape[0]
    decoded_frame_index = []
    print(f"match_video_name: {match_video_name}")
    # set batch size = 1, avoid the tedious mask
    sentence_list_dict = load_json(sentence_list_json)

    for batch_idx in range(batch_size):
        current_video_name = match_video_name[batch_idx]
        sentence_number_list = sentence_list_dict[current_video_name] # the sentence list for this particular video
        history_dict = {}
        for i in range(total_frames_to_decode):
            if i == 0:
                # last_embedding = output_embedding[batch_idx, i]
                nn_idx, total_frame_in_video = nearest_neighbour(
                    beam_size_former,  # left beam search
                    output_embedding[batch_idx, i].unsqueeze(0), 
                    current_video_name, 
                    body_contents_dir
                )
                # construct a history_dict
                for h in range(len(nn_idx)):
                    history_dict[h] = [nn_idx[h]] # in descedning order

            else: # only have one past value:
                #num_frame_to_select, last_selected_embedding, current_embedding, beam_width, match_video_name, body_contents_dir
                out_dict,total_frame_in_video = beam_search_history(
                    i,
                    beam_size_former, 
                    history_dict, # last top k indices, in desening order, this index is in body contents,
                    output_embedding[batch_idx, i].unsqueeze(0), 
                    beam_size_later,
                    lamda, 
                    current_video_name,
                    body_contents_dir,
                    sentence_number_list   
                )
                history_dict = out_dict
                # print(f"i = {i}, history_dict = {len(history_dict[0])}")
        # the first one returned by history_dict is the highest value pair
        decoded_frame_index = history_dict[0]

        # print(f"shape of decoded_list = {decoded_frame_index}")
        decoded_frame_index = np.array(decoded_frame_index)
        print(f"history_dict={decoded_frame_index.shape}")
        output_path = output_dir / f"{current_video_name}_beam_search.npy"
        np.save(output_path, decoded_frame_index)
        decoded_frame_index = consecutive_to_interval(decoded_frame_index, current_video_name, total_frame_in_video)
        print(f"decoed_frame = {decoded_frame_index.shape}")
        output_path = output_dir / f"{current_video_name}_beam_search_processed.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, decoded_frame_index)

    return decoded_frame_index

## in transformer_decode_queue.py


def find_scene_boundary(body_content_scene_arr, target):
    index = bisect.bisect_right(body_content_scene_arr, target)
    if index == 1 or index == len(body_content_scene_arr):
        return None  # Target is out of the bounds of the list
    return body_content_scene_arr[index - 1], body_content_scene_arr[index] # find the scene range

def consecutive_to_interval(arr, video_name, total_frame_in_video, scene_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/scene_main")):
    # Convert input list to a NumPy array
    # remove black first
    arr = np.array(arr)
    body_content_scene = scene_dir / video_name[0] / video_name / f"scene.npy"
    body_content_scene_arr = np.load(body_content_scene)
    # arr = remove_all_black(arr, video_name)
    # Find where the values change
    change_indices = np.where(arr[:-1] != arr[1:])[0] + 1
    # Append the start (0) and end indices of the array
    change_indices = np.concatenate(([0], change_indices, [len(arr)]))

    result = np.copy(arr)

    for i in range(len(change_indices) - 1):
        start, end = change_indices[i], change_indices[i + 1]
        if end - start > 1:  # Check if there are consecutive duplicates
            # start is the index, arr[start] is the value, start of a scene
            target = arr[start]
            # find the scene range for that frame number
            scene_start, scene_end = find_scene_boundary(body_content_scene_arr, target) # ideally, the image within one scene should the same embedding, then I only need to find the scene
            # should be convert to int
            scene_start, scene_end = int(scene_start), int(scene_end)
            scene_diff = scene_end - scene_start

            result[start:end] = np.arange(scene_start, scene_start+end-start) # either turnctae or extend
    result = np.clip(result, 0, total_frame_in_video-1) # ensure no negative value --> start with all 0 most likely be balck
    return result

if __name__ == "__main__":
    # test one 
    input_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_clip_frame")
    pass
