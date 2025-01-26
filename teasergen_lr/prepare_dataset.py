# This is used for preparing dataset
# loop through training data
import json
from pathlib import Path
import torch
import bisect
from utils import save_json, load_json, save_txt, load_txt
import os
from transformers import AutoTokenizer, CLIPModel, AutoProcessor
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torchvision.transforms import Compose, Resize, Normalize, CenterCrop, ToTensor
from transformers import RobertaModel, RobertaTokenizer
import torch
import pandas as pd
import clip
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
def get_text_features(text, model_name, device = "cuda"):
    # tokenizer = RobertaTokenizer.from_pretrained(model_name)
    # model = RobertaModel.from_pretrained(model_name)
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    model.to(device)
    # Tokenize and encode the input text
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    inputs = inputs.to(device)
    # Forward pass through the model to get hidden states
    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
    features = outputs.cpu().numpy()
    #     hidden_states = outputs.last_hidden_state

    # # To get a fixed-size representation (e.g., 512 dimensions), you can:
    # # 1. Pool the last hidden state (e.g., take the mean of all tokens)
    # features = hidden_states.mean(dim=1)
    # features = features.cpu().numpy()
    print(f"features shape: {features.shape}")
    return features

def generate_data_intro(input_dir, output_file, input_json_file, output_json_file, device="cuda"):
    # input_dir = Path("/home/weihanx/videogpt/workspace/start_code/eval/training_0905.txt")
    video_name_list = load_txt(input_dir)
    # print(f"video_name_list: {video_name_list}")
    # json_file = Path("/home/weihanx/videogpt/deepx_data6/dataset/selected_annotation_old_nopad.jsonl")
    input_json_file = Path("/home/weihanx/videogpt/deepx_data6/dataset/annotation_0904.jsonl")
    # List to store the loaded data
    data = []

    output_data = []
    # Open the file and load each line as a separate JSON object
    with open(input_json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    for data_dict in data:
        new_dict = {}
        for key, value in data_dict.items():
            video_name = key # 1iuQjyMP8_c_0
            video_name_string = video_name.rsplit('_', 1)[0] # 1iuQjyMP8_c
            query_name_string = video_name.rsplit('_', 1)[1] # 0 # TODO: they are not in order
            # print(f"type of video = {video_name_string}, type of video_name_list[0] = {video_name_list[0]}")
            
            frame_list = value['saliency']

            sentence_list = [query_name_string for i in frame_list if i == 1] # which sentence it is refering to

            # Create a list of indices where the value in frame_list is 1
            index_list = [index for index, value in enumerate(frame_list) if value == 1]
            new_dict = {
                "video": video_name,
                "sentences": sentence_list,
                "frame_indices": index_list
            }
            if new_dict not in output_data and video_name_string in video_name_list:
                output_data.append(new_dict)


    output_json_file = Path("/home/weihanx/videogpt/workspace/transformer_prior/train_0915_2.jsonl") # Replace with your desired output file path

    with open(output_json_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')

    print(f"Processed data saved to {output_json_file}")

# def generate_valid_data_intro(input_dir, device="cuda"): 
    """
    Those in introduction are treated as saliency besides sentence and frame mapping
    """
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/dataset/qid_train_video.txt")
    video_name_list = load_txt(input_dir)
    print(f"video_name_list: {video_name_list}")
    # json_file = Path("/home/weihanx/videogpt/deepx_data6/dataset/selected_annotation_old_nopad.jsonl")
    json_file = Path("/home/weihanx/videogpt/deepx_data6/dataset/annotation_0904.jsonl")
    # List to store the loaded data
    data = []

    output_data = []
    # Open the file and load each line as a separate JSON object
    with open(json_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    for data_dict in data:
        new_dict = {}
        for key, value in data_dict.items():
            video_name = key # 1iuQjyMP8_c_0
            video_name_string = video_name.rsplit('_', 1)[0] # 1iuQjyMP8_c
            query_name_string = video_name.rsplit('_', 1)[1] # already check: 
            # print(f"type of video = {video_name_string}, type of video_name_list[0] = {video_name_list[0]}")
            
            frame_list = value['saliency']

            sentence_list = [query_name_string for i in frame_list if i == 1]

            # Create a list of indices where the value in frame_list is 1
            index_list = [index for index, value in enumerate(frame_list) if value == 1]
            new_dict = {
                "video": video_name,
                "sentences": sentence_list,
                "frame_indices": index_list
            }
            # solve the problem of repetitive in 0904
            if new_dict not in output_data and video_name_string in video_name_list:
                output_data.append(new_dict)

    # output_file = Path("/home/weihanx/videogpt/workspace/transformer_prior/train.jsonl") # Replace with your desired output file path
    output_file = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_0915_2.jsonl")
    with open(output_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')

    print(f"Processed data saved to {output_file}")


def organize_dictionaries( input_file, output_file):
    """
    organize by video after generate training and validation data
    """
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    organized_dict = {}

    for item in data:
        # Extract the common prefix from the video name
        prefix = "_".join(item["video"].split("_")[:-1]) # split at last
        
        if prefix not in organized_dict:
            organized_dict[prefix] = []
        
        # Append the original dictionary under the correct prefix
        organized_dict[prefix].append(item)
    # output_file = Path( "/home/weihanx/videogpt/workspace/transformer_prior/train_comb.json")
    with open(output_file, 'w') as f:
        json.dump(organized_dict, f)

    print(f"Processed data saved to {output_file}")

    return organized_dict

def generate_train_main(file_name_path, output_dir , screen_play_folder):
    # output annotation
    """
    {"video": "rslUfl6OsV4_0", "sentences": ["0", "0", "0", "0", "0", "0", "0", "0", "0", "0"], "frame_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}
    video_name_list: main_text_name.txt: should include the name and query name
    """
    output_data = []
    video_name_list = load_txt(file_name_path)
    for vd in video_name_list:
        video_name = vd # 1iuQjyMP8_c_0
        video_name_string = video_name.rsplit('_', 1)[0] # 1iuQjyMP8_c
        query_name_string = video_name.rsplit('_', 1)[1] # 0 # TODO: they are not in order
        # load csv file and find corresponding frame indices
        print(f"video_name_string = {video_name_string}, query_name_string = {query_name_string}")
        csv_file = screen_play_folder / video_name_string[0]/ video_name_string / "script_timeline.csv"
        # index by query name string
        timeline_df = pd.read_csv(csv_file)
        search_index = int(query_name_string)
        start_time = timeline_df.iloc[search_index]['Start Time']
        end_time = timeline_df.iloc[search_index]['End Time']
        print(f"start_time = {start_time}, end_time = {end_time}")
        frame_start = round(start_time)
        frame_end = round(end_time)
        frame_indices = list(range(frame_start, frame_end+1))
        sentence_list = [query_name_string for i in frame_indices]
        new_dict = {
            "video": video_name,
            "sentences": sentence_list,
            "frame_indices": frame_indices
        }
        output_data.append(new_dict)
    # save list
    output_json_file = output_dir / "train_0103_intro.jsonl" # Replace with your desired output file path
    with open(output_json_file, 'w') as f:
        for item in output_data:
            f.write(json.dumps(item) + '\n')

    print(f"Processed data saved to {output_json_file}")   

def load_frames_from_video(video_dir):
    """Load frames from a given video directory."""
    frame_files = sorted([f for f in video_dir.iterdir() if f.is_file()])
    frames = [Image.open(video_dir/f) for f in frame_files]
    return frames



def apply_clip_to_frames(frames, model, processor, device="cuda"):
    embeddings = []
    model = model.to(device)

    for frame in frames:
        inputs = processor(images=frame, return_tensors="pt").to(device)  # Process the frame with processor
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
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.eval()

    # Define the transform to be applied to each frame


    # Process each video in the video_name_list
    for video_name in video_name_list:
        video_dir = input_dir / video_name[0] / video_name
        if not video_dir.exists():
            print(f"Video directory {video_dir} not found, skipping...")
            fail_dataset.append(video_name)
            continue
        
        # Load frames from the video directory
        frames = load_frames_from_video(video_dir)
        # Apply ResNet to each frame to generate embeddings
        embeddings = apply_clip_to_frames(frames, model, processor)
        print(f"embeddings shape: {embeddings.shape} and it has {len(frames)} frames") # 132, 2048, 1, 1
        # Save the embeddings to the output directory 
        output_path = output_dir / f"{video_name}_clip.npy"
        # output_path = os.path.join(output_dir, f"{video_name}_embeddings.npy")
        
        save_embeddings(embeddings, output_path)
        # break
    output_path = output_dir / "fail_dataset.txt"
    with open(output_path, "w") as f:
        for item in fail_dataset:
            f.write("%s\n" % item)

def generate_image_emb_l14(input_dir, video_name_path, output_dir, device = "cuda"):
    video_name_list = load_txt(video_name_path) 
    # Load the pre-trained ResNet model
    fail_dataset = []
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-large-patch14")


    model.eval()

    # Define the transform to be applied to each frame


    # Process each video in the video_name_list
    for video_name in video_name_list:
        video_dir = input_dir / video_name[0] / video_name
        if not video_dir.exists():
            print(f"Video directory {video_dir} not found, skipping...")
            fail_dataset.append(video_name)
            continue
        
        # Load frames from the video directory
        frames = load_frames_from_video(video_dir)
        # Apply ResNet to each frame to generate embeddings
        embeddings = apply_clip_to_frames(frames, model, processor)
        print(f"embeddings shape: {embeddings.shape} and it has {len(frames)} frames") # 132, 2048, 1, 1
        # Save the embeddings to the output directory 
        output_path = output_dir / f"{video_name}_clip.npy"
        # output_path = os.path.join(output_dir, f"{video_name}_embeddings.npy")
        
        save_embeddings(embeddings, output_path)
        # break
    output_path = output_dir / "fail_dataset.txt"
    with open(output_path, "w") as f:
        for item in fail_dataset:
            f.write("%s\n" % item)


# Example usage:
# generate_image_emb('./frames', ['video1', 'video2'], './embeddings')
def generate_gpt_text_emb(video_name_path, input_dir, output_dir, device="cuda"):
    video_name_list = load_txt(video_name_path)
    input_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824")
    output_dir = Path("/home/weihanx/videogpt/workspace/clip_gpt_text_emb")
    for vn in video_name_list:
        # load csv

        csv_file = input_dir / vn[0] / vn / "story_script.csv"
        df = pd.read_csv(csv_file)
        for row, content in df.iterrows():
            text = content['text']
            id = content['id']
            features = get_text_features(text, model_name = "openai/clip-vit-base-patch32")
            print(f"features shape: {features.shape}")
            save_path = output_dir / f"{vn}_{id}.npy"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(save_path, features)
    # pass # the text comes from the generated text: /home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824/-/-A8qdoRJbPI/narr_script.json
def generate_gpt_test_data(video_name_path, input_dir, output_path, device = "cuda"): # I will use the generated text and its length
    # the length of sentence is:  /home/weihanx/videogpt/deepx_data6/gpt_demo/speech_0825/U/UzivaxYf1iM/speech_duration.json
    video_name_list = load_txt(video_name_path)
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/speech_0825")
    # output_path = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt.json")
    output_data = {}
    for vn in video_name_list:
        # load csv
        output_data[vn] = []
        length_file = input_dir / vn[0] / vn / "speech_duration.json"
        length = load_json(length_file)
        
        for key, value in length.items():
            id = key
            duration = math.ceil(value)
            sentences = [id for i in range(duration)]
            new_dict = {
                "video": f"{vn}_{id}",
                "sentences": sentences,
                "frame_indices": list(range(duration)) # useless, I don't know GT
                # "duration": duration # no need for frame index, no ground truth
            }
            output_data[vn].append(new_dict)
    save_json(output_path, output_data)

def generate_sentence_list(test_json_file, output_file): 
    """
    use for decoding
    """
    
    data = load_json(test_json_file)
    video_sentence_dict = {}
    
    # Iterate through each video in the data
    for video_id, video_data in data.items():
        # Initialize an empty list to store concatenated sentences for each video
        concatenated_sentences = []
        
        # Iterate through all entries (video segments) for each video name
        for entry in video_data:
            # Concatenate the sentences for the current video segment
            concatenated_sentences += entry["sentences"]
        
        # Store the concatenated list in the dictionary with video_name as the key
        video_sentence_dict[video_id] = concatenated_sentences
    save_json(output_file, video_sentence_dict)
    return video_sentence_dict

def label_frames(scene_starts):
    # Assume the last element in scene_starts is the total number of frames
    total_frames = scene_starts[-1]
    scene_starts = scene_starts[:-1]  # Remove the last element as it's not a scene start, it just indicates the total number of frames

    # Initialize the list to store the scene labels for each frame
    scene_labels = []
    total_frames = int(total_frames)
    # Loop through all frames from 0 to total_frames - 1
    for frame_index in range(total_frames):
        # Use bisect to find the right scene by finding where the frame would fit in the scene_starts list
        # minus 1 because bisect gives the position this element would be inserted to achieve a sorted list
        # and we need the index of the previous element as it denotes the scene change
        scene_index = bisect.bisect_right(scene_starts, frame_index) - 1
        scene_labels.append(scene_index)

    return scene_labels


# generate scene list for each video 
def extract_scene_list(video_name_path, input_dir, input_narr_dir, output_dir):
    """
    output generation: .json 
    """
    # json_file = Path("/home/weihanx/videogpt/workspace/transformer_prior/train_sentence_list.json") # find the scene number for those text

    # json_file_list = load_json(json_file)
    video_name_list = load_txt(video_name_path)
    for vn in video_name_list:
        # print(f"vn = {vn}")
        # json_file_list_len = len(json_file_list[vn])
        # print(f"json_file_list = {len(json_file_list)}") # need to make sure each frame has one scene labbel
        scene_path = input_dir / vn[0] / vn / "scene.npy" # scene should be the same length as the actual video
        # for each sentence list, I will have corresponding scene list
        scene_arr = np.load(scene_path) # include 0 at the beginning and total length at the end
        # first generate a long list denote the scene group
        # need to know how I get sentence embeddings, how to round the upper and end 
        scene_labels = label_frames(scene_arr) # return a list, start from 0 second
        # print(f"scene_labels = {len(scene_labels)}")
        narration_script = input_narr_dir / vn[0] / vn / "ts_comb.csv"
        df = pd.read_csv(narration_script)
        scene_anno = []
        scene_count = 0
        for index, row in df.iterrows():
            start_clip = round(row['start'])
            end_clip = math.lower(row['end']) # avoid last one take upper value
            len_of_scene = end_clip - start_clip + 1
            scene_anno.extend([scene_count]*len_of_scene)
            # scene_anno.extend(scene_labels[start_clip:end_clip+1])
        # save numpy
            scene_count += 1 # next scene
        # print(f"scene_anno = {len(scene_anno)}")
        output_path = output_dir / vn[0] / vn / "scene_cut.npy"
        # assert len(scene_anno) == json_file_list_len, "scene_anno and scene_labels should have the same length"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, scene_anno)
            
        # then extract the timestamp with narrations
        # load the sentence list
        

if __name__ == "__main__":
    pass