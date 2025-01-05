# load image
# feature extraction for all subclips in test set
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from transformers import CLIPProcessor, CLIPModel, AutoProcessor
import json
from pathlib import Path
import json
import math
import torch
from helper import save_json, load_json, save_txt, load_txt, load_csv
import os
import matplotlib.pyplot as plt
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
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import clip
import re
model, preprocess = clip.load("ViT-B/32", device="cuda:3")
def apply_clip_to_frames(frames, model, processor, device="cuda:3"):
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

def load_frames_from_video(video_dir):
    """Load frames from a given video directory."""
    frame_files = sorted([f for f in video_dir.iterdir() if f.is_file()])
    frames = [Image.open(video_dir/f) for f in frame_files]
    return frames

def generate_image_emb(input_dir, video_name_path, output_dir, device = "cuda:3"):
    video_name_list = load_txt(video_name_path) 
    # Load the pre-trained ResNet model
    fail_dataset = []
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.eval()
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
        output_path = output_dir / video_name[0] / video_name /  f"{video_name}_demo.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # output_path = os.path.join(output_dir, f"{video_name}_embeddings.npy")
        
        save_embeddings(embeddings, output_path)
        # break
    output_path = output_dir / "fail_dataset.txt"
    with open(output_path, "w") as f:
        for item in fail_dataset:
            f.write("%s\n" % item)

def get_similarity_score(video_name, time_interval_dir, body_contents_dir, time_interval_name, gpt):
    # frame_embedding: # total_frames, 512
    # sentence_embedding: # total_sentences, 512
    # sentences: /home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824/-/-A8qdoRJbPI/story_script.csv
    # range: /home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_tf/-/-A8qdoRJbPI/time_interval_min_3.json
    """
    Goal: Get average cosine similarity between text and image of video_name
    """
    print(f"video_name: {video_name}")
    frame_emb = []
    for i in range(10): # I divide them into 10 chunks for each video clip
        image_emb_path = body_contents_dir/ video_name[0] / f"{video_name}_clip_{i}" / f"{video_name}_clip_{i}_body_clip.npy" # load all test video and do inference
        image_emb = np.load(image_emb_path)
        frame_emb.append(image_emb)
    combined_frame_emb = np.vstack(frame_emb)
    # print(f"combined_frame_emb shape: {combined_frame_emb.shape}")
    interval_path = time_interval_dir / video_name[0] / video_name / f"{time_interval_name}.json"
    # interval_path = time_interval_dir / f"{video_name}_threshold.json"
    intervals = load_json(interval_path)
    print(f"intervals = {intervals}")
    # print(f"intervals: {intervals.keys()}")
    # print(f"script_csv shape: {script_csv.shape}, content: {script_csv}")   

    long_cosine_similarities_list = []
    if gpt == False:
        feature_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
    else:
        # feature_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_clip_sentence_emb_512_average_gpt")
        feature_dir = Path("/home/weihanx/videogpt/deepx_data7/clip_sentence_emb_512_average_gpt")
    for key, ranges_list in intervals.items():
        if video_name == "2n7xgho1gIo" and int(key) == 17:
            continue # this video name should not be included, junk data
        for t in ranges_list:
            start = t[0]
            end = t[1]
            # not integer for random :
            start = round(start)
            end = round(end)
            print(f"start = {start}, end = {end}")
            cur_image_emb = combined_frame_emb[start:end+1] # inclusive
            row_id = int(key) # start from zero
            print(f"row_id = {row_id}, start = {start} , end = {end}")
            text_emb_path = feature_dir / f"{video_name}_{row_id}.npy"
            text_emb = np.load(text_emb_path)
        
            cosine_similarities = cosine_similarity(text_emb, cur_image_emb).flatten() # shape: (N_frames,)

        # average over list
            long_cosine_similarities_list.extend(cosine_similarities)
    long_cosine_similarities_array = np.array(long_cosine_similarities_list)
    print(f"shape of overall cosine similarty = {long_cosine_similarities_array.shape}")
    save_dir =  time_interval_dir / "clip_score"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{video_name}_cosine_similarity.npy"
    np.save(save_path, long_cosine_similarities_array)
    average_cos = np.mean(long_cosine_similarities_array)
    return average_cos
        # calculate the average embedding

def get_similarity_score_s(video_name, time_interval_dir, body_contents_dir, time_interval_name, gpt):
    # frame_embedding: # total_frames, 512
    # sentence_embedding: # total_sentences, 512
    # sentences: /home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824/-/-A8qdoRJbPI/story_script.csv
    # range: /home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_tf/-/-A8qdoRJbPI/time_interval_min_3.json
    """
    Goal: Get average cosine similarity between text and image of video_name
    """
    print(f"video_name: {video_name}")
    frame_emb = []
    for i in range(10): # I divide them into 10 chunks for each video clip
        image_emb_path = body_contents_dir/ video_name[0] / f"{video_name}_clip_{i}" / f"{video_name}_clip_{i}_body_clip.npy" # load all test video and do inference
        image_emb = np.load(image_emb_path)
        frame_emb.append(image_emb)
    combined_frame_emb = np.vstack(frame_emb)
    # print(f"combined_frame_emb shape: {combined_frame_emb.shape}")
    interval_path = time_interval_dir / video_name[0] / video_name / f"{time_interval_name}.json"
    # interval_path = time_interval_dir / f"{video_name}_threshold.json"
    intervals = load_json(interval_path)
    print(f"intervals = {intervals}")
    # print(f"intervals: {intervals.keys()}")
    # print(f"script_csv shape: {script_csv.shape}, content: {script_csv}")   
    # cosine_similarities_list = []
    long_cosine_similarities_list = []
    if gpt == False:
        feature_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
    else:
        # feature_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_clip_sentence_emb_512_average_gpt")
        feature_dir = Path("/home/weihanx/videogpt/deepx_data7/clip_sentence_emb_512_average_gpt")
    for key, ranges_list in intervals.items():
        if video_name == "2n7xgho1gIo" and int(key) == 17:
            continue # this video name should not be included, junk data
        for t in ranges_list:
            start = t[0]
            end = t[1]
            # not integer for random :
            start = round(start)
            end = round(end)
            print(f"start = {start}, end = {end}")
            cur_image_emb = combined_frame_emb[start:end+1] # inclusive
            row_id = int(key) # start from zero
            print(f"row_id = {row_id}, start = {start} , end = {end}")
            text_emb_path = feature_dir / f"{video_name}_{row_id}.npy"
            text_emb = np.load(text_emb_path)
        
            cosine_similarities = cosine_similarity(text_emb, cur_image_emb).flatten() # shape: (N_frames,)
            # print(f"cosine_similarities content: {cosine_similarities}")
            # clip negative value
            cosine_similarities = np.clip(cosine_similarities, 0, None)

        # average over list
            long_cosine_similarities_list.extend(cosine_similarities)
    long_cosine_similarities_array = np.array(long_cosine_similarities_list) * 2.5 # scale up
    print(f"shape of overall cosine similarty = {long_cosine_similarities_array.shape}")
    save_dir =  time_interval_dir / "clip_score_s"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{video_name}_cosine_similarity.npy"
    np.save(save_path, long_cosine_similarities_array)
    average_cos = np.mean(long_cosine_similarities_array)
    return average_cos

# def get_similarity_score_old(video_name, time_interval_dir, story_script_dir, frame_emb_input_dir, time_interval_name, csv_file_name, gpt):
#     # frame_embedding: # total_frames, 512
#     # sentence_embedding: # total_sentences, 512
#     # sentences: /home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824/-/-A8qdoRJbPI/story_script.csv
#     # range: /home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_tf/-/-A8qdoRJbPI/time_interval_min_3.json
#     """
#     Goal: Get average cosine similarity between text and image of video_name
#     """
#     print(f"video_name: {video_name}")
#     image_embedding_path = frame_emb_input_dir / video_name[0] / video_name / f"{video_name}_demo.npy"
#     image_emb = np.load(image_embedding_path)

#     interval_path = time_interval_dir / video_name[0] / video_name / f"{time_interval_name}.json"
#     intervals = load_json(interval_path)
#     print(f"intervals: {intervals.keys()}")
#     script_csv_path = story_script_dir / video_name[0] / video_name / f"{csv_file_name}.csv"
#     script_csv = pd.read_csv(script_csv_path)
#     # print(f"script_csv shape: {script_csv.shape}, content: {script_csv}")   
#     current_start = 0
#     ranges = {}
#     long_cosine_similarities_list = []
#     for key, ranges_list in sorted(intervals.items(), key=lambda item: int(item[0])):
#         if video_name == "2n7xgho1gIo" and int(key) == 17:
#             continue
#         print(f"Key: {int(key)}, Type: {type(key)}") 
#         # fine the text in csv files
#         # print(f"key: {key}, type(key): {type(key)}")

#         total_length = sum((round(end) - round(start)) + 1 for start, end in ranges_list)
#         current_end = current_start + total_length - 1
#         ranges[key] = [current_start, current_end] # range: inclusive!!
#         # load the image embedding
#         # print(f"key = {key}")
#         if gpt == False:
#             feature_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
#         else:
#             feature_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
#         row_id = int(key) # start from zero
#         text_emb_path = feature_dir / f"{video_name}_{row_id}.npy"
#         text_emb = np.load(text_emb_path)
        
#         cur_image_emb = image_emb[current_start:current_end+1] # inclusive
#         # cosine similarity between text and image
#         # avergae cosine similiarty
#         # print (f"text_emb shape: {text_emb.shape}, cur_image_emb shape: {cur_image_emb.shape}")
#         cosine_similarities = cosine_similarity(text_emb, cur_image_emb) 
#         # print(f"cosine_similarities shape: {cosine_similarities.shape}, content: {cosine_similarities}")

#         current_start = current_end + 1 # inclusive, should be int
#     # average over list
#         long_cosine_similarities_list.extend(cosine_similarities.flatten()) # all cosine simiarlty
#     long_cosine_similarities_array = np.array(long_cosine_similarities_list)
#     save_dir =  Path("/home/weihanx/videogpt/workspace/UniVTG/clip_score_gpt_ft")
#     save_dir.mkdir(parents=True, exist_ok=True)
#     save_path = save_dir / f"{video_name}_cosine_similarity.npy"
#     np.save(save_path, long_cosine_similarities_array)
#     average_cos = np.mean(long_cosine_similarities_array, axis=0)
    
#     return average_cos
#         # calculate the average embedding
# def get_similarity_score_fps4(video_name, time_interval_dir, story_script_dir, frame_emb_input_dir, time_interval_name, csv_file_name, gpt):
#     """
#     Goal: Get average cosine similarity between text and image of video_name.
#     Now adjusted for fps=4.
#     """
#     fps = 4  # Frames per second
#     print(f"video_name: {video_name}")
    
#     # Load image embeddings
#     image_embedding_path = frame_emb_input_dir / video_name[0] / video_name / f"{video_name}_demo.npy"
#     image_emb = np.load(image_embedding_path)
#     print(f"image_emb shape: {image_emb.shape}")
#     # Load time intervals
#     interval_path = time_interval_dir / video_name[0] / video_name / f"{time_interval_name}.json"
#     intervals = load_json(interval_path)
#     print(f"intervals: {intervals.keys()}")

#     # Load script CSV
#     script_csv_path = story_script_dir / video_name[0] / video_name / f"{csv_file_name}.csv"
#     script_csv = pd.read_csv(script_csv_path)

#     current_start = 0  # Initialize frame-based start
#     ranges = {}
#     cosine_similarities_list = []
#     long_cosine_similarities_list = []

#     for key, ranges_list in sorted(intervals.items(), key=lambda item: int(item[0])):
#         if video_name == "2n7xgho1gIo" and int(key) == 17:
#             continue
#         print(f"Key: {int(key)}, Type: {type(key)}")

#         # Calculate the total length of time in seconds, and then convert to frames
#         total_length_seconds = sum((round(end) - round(start)) + 1 for start, end in ranges_list)
#         total_frames = total_length_seconds * fps  # Convert seconds to frames

#         current_end = current_start + total_frames - 1  # Frame-based end
#         ranges[key] = [current_start, current_end]  # Store frame-based ranges

#         # Load the sentence embedding
#         feature_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average") if not gpt else Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average_gpt")
#         row_id = int(key)  # Start from zero
#         text_emb_path = feature_dir / f"{video_name}_{row_id}.npy"
#         text_emb = np.load(text_emb_path)
#         print(f"video_name = {video_name}, total_frames: {total_frames}")
#         # Process frame-based embedding in intervals of fps=4 frames per second
#         for second in range(total_length_seconds):
#             # Define the frame range for this second (4 frames per second)
#             frame_start = current_start + second * fps
#             frame_end = frame_start + fps  # Get 4 frames for the current second
#             print(f"Frame start: {frame_start}, Frame end: {frame_end}")
#             # Extract the image embeddings for these frames
#             cur_image_emb = image_emb[frame_start:frame_end]  # Shape will be (4, embedding_dim)
#             print(f"cur_image_emb shape: {cur_image_emb.shape}, text_emb shape: {text_emb.shape}")
#             # Calculate cosine similarities between text embedding and these frame embeddings
#             cosine_similarities = cosine_similarity(text_emb.reshape(1, -1), cur_image_emb)

#             # Get the highest cosine similarity within these 4 frames
#             max_cosine_similarity = np.max(cosine_similarities)
#             cosine_similarities_list.append(max_cosine_similarity)  # Store max similarity for this second

#             # Append all frame similarities for long array (for detailed logging or other purposes)
#             long_cosine_similarities_list.extend(cosine_similarities.flatten())

#         # Update current_start for the next segment (frame-based indexing)
#         current_start += total_frames

#     # Convert long similarity list to array and save
#     long_cosine_similarities_array = np.array(long_cosine_similarities_list)
#     save_dir = Path("/home/weihanx/videogpt/workspace/UniVTG/demo_ft_clip_score_fps4")
#     save_dir.mkdir(parents=True, exist_ok=True)
#     save_path = save_dir / f"{video_name}_cosine_similarity.npy"
#     np.save(save_path, long_cosine_similarities_array)

#     # Calculate the average cosine similarity over all seconds
#     cosine_similarities_array = np.array(cosine_similarities_list)
#     average_cos = np.mean(cosine_similarities_array, axis=0)
    
#     return average_cos
# def get_similarity_score_intro(video_name, story_script_dir, frame_emb_input_dir, csv_file_name):
#     # frame_embedding: # total_frames, 512
#     # sentence_embedding: # total_sentences, 512
#     # sentences: /home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824/-/-A8qdoRJbPI/story_script.csv
#     # range: /home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_tf/-/-A8qdoRJbPI/time_interval_min_3.json
#     """
#     Goal: Get average cosine similarity between text and image of video_name
#     """
#     print(f"video_name: {video_name}")
#     image_embedding_path = frame_emb_input_dir / video_name[0] / video_name / f"{video_name}_actual_demo.npy"
#     image_emb = np.load(image_embedding_path)
#     script_csv_path = story_script_dir / video_name[0] / video_name / f"{csv_file_name}.csv"
#     script_csv = pd.read_csv(script_csv_path)
#     # print(f"script_csv shape: {script_csv.shape}, content: {script_csv}")   
#     feature_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
#     cosine_similarities_list = []
#     long_cosine_similarities_list = []
#     # for row, data in script_csv.iterrows():
#     #     start_pt = round(data['start'])
#     #     end_pt = round(data['end'])
#     #     row_id  = data['id']
#     #     # load from previous id
#     #     text_emb_path = feature_dir / f"{video_name}_{row_id}.npy"
#     #     text_emb = np.load(text_emb_path)
#     #     cur_image_emb = image_emb[start_pt:end_pt+1] # inclusive
#     #     cosine_similarities = cosine_similarity(text_emb, cur_image_emb) 
#     #     print(f"cosine_similarities shape: {cosine_similarities.shape}, content: {cosine_similarities}")
#     #     long_cosine_similarities_list.append(cosine_similarities)   
#     #     average_cos_clip = np.mean(cosine_similarities) # average over sentences
#     #     cosine_similarities_list.append(average_cos_clip)

#     # Iterate over each row in the CSV
#     for row, data in script_csv.iterrows():
#         start_pt = round(data['start'])
#         end_pt = round(data['end'])
#         row_id  = data['id']
        
#         # Load the text embedding for the current sentence
#         text_emb_path = feature_dir / f"{video_name}_{row_id}.npy"
#         text_emb = np.load(text_emb_path)
        
#         # Get the image embeddings corresponding to the frames in this segment
#         cur_image_emb = image_emb[start_pt:end_pt+1]  # inclusive, cur_image_emb shape (N_frames, 512)
        
#         # Reshape text_emb to (1, 512) for compatibility with cur_image_emb (N_frames, 512)
#         text_emb = text_emb.reshape(1, -1)
        
#         # Compute cosine similarities between the text embedding and each frame embedding
#         cosine_similarities = cosine_similarity(text_emb, cur_image_emb).flatten()  # Shape (N_frames,)
        
#         # Append this segment's cosine similarities to the long list
#         long_cosine_similarities_list.extend(cosine_similarities)

#     # Convert the long list to a numpy array for easier manipulation or saving if needed
#     long_cosine_similarities_array = np.array(long_cosine_similarities_list)

#         # average over list
#     # cosine similarity for one video when fps is 1
#     save_dir = Path("/home/weihanx/videogpt/workspace/UniVTG/clip_score_fps8")
#     save_dir.mkdir(parents=True, exist_ok=True)
#     save_path = save_dir / f"{video_name}_cosine_similarity.npy"
#     np.save(save_path, long_cosine_similarities_array)
#     cosine_similarities_list = np.array(cosine_similarities_list) # average over whole clips
#     average_cos = np.mean(cosine_similarities_list, axis=0)
#     return average_cos


def get_overall_similarity_score(video_name_path, time_interval_dir, body_contents_dir,time_interval_name, gpt):
    # get all video names
    video_names = load_txt(video_name_path)
    overall_cosine_similarities = []
    buggy_video = []
    df = pd.DataFrame(columns=["video_name", "average_cos"])
    for video_name in video_names:
        # if video_name!="j1Pb8YU8wQo":
            # continue
        average_cos = get_similarity_score_s(video_name, time_interval_dir, body_contents_dir, time_interval_name, gpt)
        average_cos_s = get_similarity_score(video_name, time_interval_dir, body_contents_dir, time_interval_name, gpt)
        # print(f"video_name: {video_name}, average_cos: {average_cos}")
        # # # try:
        # # if video_name != "nTq1Sd9N7E8":
        # #     continue
        # if type == "intro":
        #     if fps == 4:
        #         average_cos = get_similarity_score_fps4(video_name, time_interval_dir, story_script_dir, frame_emb_input_dir, time_interval_name, csv_file_name, gpt)
        #     else:
        #         average_cos = get_similarity_score_intro(video_name, story_script_dir, frame_emb_input_dir, csv_file_name)
        # if type == "gen":
        #     if fps == 4:
        #         average_cos = get_similarity_score_fps4(video_name, time_interval_dir, story_script_dir, frame_emb_input_dir, time_interval_name, csv_file_name, gpt)
        #     else:
        #         average_cos = get_similarity_score(video_name, time_interval_dir, story_script_dir, frame_emb_input_dir,time_interval_name, csv_file_name, gpt)
        row = {"video_name": video_name, "average_cos": average_cos, "average_cos_s": average_cos_s}
        df = df.append(row, ignore_index=True)
        # except Exception as e:
        #     print(f"Error: {e} video_name: {video_name}")
        #     buggy_video.append(video_name)
    df.to_csv(time_interval_dir/"overall_similarity_scores.csv", index=False)
    
    print(df.mean())

def plot_clip_score(input_dir, video_name, output_dir):
    # Load the clip score
    clip_score_path = input_dir / f"{video_name}_cosine_similarity.npy"
    clip_score = np.load(clip_score_path)
    print(f"clip_score shape: {clip_score.shape}")

    # Plot the clip score
    plt.figure()  # Create a new figure
    plt.plot(clip_score)
    plt.xlabel("Frame Index")
    plt.ylabel("Cosine Similarity")
    plt.title(f"Clip Score for {video_name}")
    
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_name}_clip_score.png"
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()  # Close the plot to avoid multiple lines overlapping

def check_if_text_too_long(text, model, max_token_length=77):
    """Check if the text is too long for CLIP's tokenization limit."""
    try:
        tokenized_text = clip.tokenize(text)
        return False
    except Exception as e:
        # print(f"Error tokenizing text: {e}")
        return True

def split_text_by_punctuation(text):
    """Split text by punctuation marks (e.g., period, comma, etc.)."""
    # Use regex to split by common punctuation marks (period, comma, exclamation mark, question mark, semicolon, colon, etc.)
    sentences = re.split(r'[，。！？,.;:]', text)
    # Filter out empty strings
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    return sentences

# def split_text_into_sentences(text):
#     """Splits text into sentences using NLTK."""
#     sentences = sent_tokenize(text)
#     return sentences
def process_text_with_clip(text, model):
    """Get the CLIP embedding for a given text."""
    # Tokenize text and print token ids to check for out-of-range values
    text_inputs = clip.tokenize(text).to("cuda:3")

    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
    return text_features.cpu().numpy()

def get_gt_clipscore():
    # load frame embedding
    fail_video = []
    frame_embedding_dir = Path("/home/weihanx/videogpt/workspace/clip_image_emb_full")
    video_name_path = Path("/home/weihanx/videogpt/0719_data_name.txt")
    video_name_list = load_txt(video_name_path)
    sentence_embedding_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_512_average")
    script_dir = Path("/home/weihanx/videogpt/deepx_data6/dataset")
    df = pd.DataFrame(columns=["video_name", "average_cos"])
    for vd in video_name_list:
        try:
            csv_file = pd.read_csv(script_dir / vd[0] / vd / "ts_comb.csv")

            frame_emb_path = frame_embedding_dir / f"{vd}_clip.npy"
            frame_emb = np.load(frame_embedding_dir / f"{vd}_clip.npy")
            long_cosine_similarities_list = []
            for i, row in csv_file.iterrows():
                start = math.ceil(row["start"])
                end = math.floor(row["end"])
                id = row["id"]
                cur_image_emb = frame_emb[start:end+1]
                print(f"cur_image_emb shape: {cur_image_emb.shape}")
                text_emb = np.load(sentence_embedding_dir / f"{vd}_{id}.npy")
                
                cosine_similarities = cosine_similarity(text_emb, cur_image_emb)
                cosine_similarities = cosine_similarities.flatten()
                cosine_similarities = np.clip(cosine_similarities, 0, None)
                cosine_similarities = 2.5 * cosine_similarities
                long_cosine_similarities_list.extend(cosine_similarities)
            df = df.append({"video_name": vd, "average_cos": np.mean(long_cosine_similarities_list)}, ignore_index=True)
        except Exception as e:
            print(f"Error: {e}, video_name: {vd}")
            fail_video.append(vd)
    df.to_csv("gt_clip_score.csv", index=False)
    print(f"average_cos: {df.mean()}")
    print(f"fail_video: {fail_video}")


def get_saliency_score(video_name_list, saliency_curve_dir, time_interval_dir, time_interval_name, output_dir):
    # load saliency score curve
    # get sentence json
    df = pd.DataFrame(columns=["Video_Name", "Saliency_Score"])
    for vd in video_name_list:
        # load time interval
        saliency_score = 0
        time_interval_path = time_interval_dir / vd[0] / vd / time_interval_name
        # time_interval_path = time_interval_dir / f"{vd}_threshold.json"
        time_intervals_dict = load_json(time_interval_path)
        saliency_score = 0
        count = 0
        for key, value in time_intervals_dict.items():
            # load saliency score 
            # # print(f"key = {key}. This is the sentence number")
            if int(key) == 17 and vd == "2n7xgho1gIo":
                continue
            saliency_score_path = saliency_curve_dir / f"{vd}_{int(key)}_saliency_curve.npy"
            saliency_score_np = np.load(saliency_score_path)
            for interval in value:
                start, end = interval
                start = round(start)
                end = round(end)
                sum_count = end - start + 1
                count += sum_count # selected frames
                saliency_score += saliency_score_np[start:end+1].sum() # include end
        
        saliency_score = saliency_score / count
                
        row = pd.DataFrame([[vd, saliency_score]], columns=df.columns)
        df = pd.concat([df, row], ignore_index=True)
    average_dict = df.mean().to_dict()
    print(f"average_dict saliency score = {average_dict}")
    df.to_csv(output_dir, index=False)

if __name__ == "__main__":
    # get_gt_clipscore()
    # input_dir  = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_random/frames")
    # # input_dir = Path("/home/weihanx/videogpt/deepx_data7/frame_outputs_main_test") # save the whole body contents

    # # output_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_random/gpt_demo_embedding") # need to separate the body and concat together for global serarch
    # # output_dir.mkdir(parents=True, exist_ok=True)
    # # generate_image_emb(input_dir, video_name_path, output_dir)

    # story_script_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824")
    # frame_emb_input_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_random/gpt_demo_embedding")
    # get_overall_similarity_score(video_name_path, time_interval_dir, story_script_dir, frame_emb_input_dir)
    # test_folder = in test,josn key
    
    # number of 1s in saliency score ==>  number of sentences --> frame index range in the predicted npy file <--> query
    # also can check rough range of predicted score

    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # ## Ground Truth ClipScore ######
    # narration_type = "gt"
    # narration_dir = Path("/home/weihanx/videogpt/deepx_data6/dataset")
    # # This index is on intro frame level
    # frame_folder = Path("frame_outputs_intro_test_fps4")
    # frame_embedding_folder = Path("frame_outputs_intro_test_fps4/embeddings")
    # frame_embedding_folder.mkdir(parents=True, exist_ok=True)
    # csv_file_name = "ts_comb"
    # # generate_image_emb(frame_folder, video_name_path, frame_embedding_folder) # get frame input
    # overall_average_cos, buggy_video = get_overall_similarity_score(video_name_path, story_script_dir, narration_dir, frame_embedding_folder, story_script_dir,csv_file_name, "intro") # get simarlity
    # print(f"model_type_folder: truth, overall_average_cos: {overall_average_cos}, buggy_video: {buggy_video}")
   

    # # input_dir = Path("/home/weihanx/videogpt/workspace/UniVTG/clip_score_fps4")
    # # output_dir = Path("/home/weihanx/videogpt/workspace/UniVTG/clip_score_fps4_plot")
    # video_name_path = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_name.txt")
    # body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip_frame_512_emb")
    # video_name_list = load_txt(video_name_path)

    # # demo_dir = Path("/home/weihanx/videogpt/deepx_data6/demo")
    gpt = True
    # demo_folder = "demo_random"
    # # eval_name = "gpt_random_0901"
    model_type_folder = "gpt_tf_1117"

    time_interval_name = "time_interval_min_3_0"

    video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    video_name_list = load_txt(video_name_path)
    body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    time_interval_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/gpt_tf_1117")
    get_overall_similarity_score(video_name_path, time_interval_dir, body_contents_dir, time_interval_name, gpt) # get simarlity

    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    # video_name_list = load_txt(video_name_path)
    # demo_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo")
    # gpt = True
    # time_interval_name = "threshold"
    # time_interval_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/transformer_threshold_gpt_0926") # demo_tf
    # get_overall_similarity_score(video_name_path, time_interval_dir, body_contents_dir, time_interval_name, gpt) # get simarlity
    # for video_name in video_name_list:
    #     plot_clip_score(input_dir, video_name, output_dir)

    # ### Demo ClipScore ######

    # narration_type = "demo"
    # # folder name: ts_comb.csv


    # ###---------- Demo_tf ClipScore ######

    




# # ### GPT evaluation #### 
    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # video_name_list = load_txt(video_name_path)
    # narration_type = "gpt"  
    # if narration_type == "gpt":
    #     narration_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo/narration_0824")
    #     # folder name: stroy_script.csv
    #     demo_dir = Path("/home/weihanx/videogpt/deepx_data6/gpt_demo")
    #     csv_file_name = "story_script"

#     ### Demo_ft clipScore ####
    # model_type_folder = "gpt_tf_1117"

    # time_interval_name = "time_interval_min_5_1.json"

    # time_interval_dir = demo_dir / model_type_folder # demo_ft
    # frame_folder = demo_dir / model_type_folder / "frames"
    # frame_emb_input_dir = demo_dir / model_type_folder / "embeddings"
    # # generate_image_emb(frame_folder, video_name_path, frame_emb_input_dir) # get frame input
    # overall_average_cos, buggy_video = get_overall_similarity_score(video_name_path, time_interval_dir, narration_dir, frame_emb_input_dir, time_interval_name, csv_file_name,"gen", True) # get simarlity
    # # print(f"model_type_folder: {model_type_folder}, overall_average_cos: {overall_average_cos}, buggy_video: {buggy_video}")
    # # input_dir = Path("/home/weihanx/videogpt/workspace/UniVTG/clip_score_gpt_ft")
    # output_dir = Path("/home/weihanx/videogpt/workspace/UniVTG/clip_score_gpt_ft_plot")
    # for video_name in video_name_list:
    #     plot_clip_score(input_dir, video_name, output_dir)
#     ###---------- Demo_tf ClipScore ######
#     model_type_folder = "gpt_tf_0825"

#     time_interval_name = "time_interval_min_3"
    
#     time_interval_dir = demo_dir / model_type_folder # demo_tf
#     frame_folder = demo_dir / model_type_folder / "frames"
#     frame_emb_input_dir = demo_dir / model_type_folder / "embeddings"
#     generate_image_emb(frame_folder, video_name_path, frame_emb_input_dir) # get frame input
#     overall_average_cos, buggy_video = get_overall_similarity_score(video_name_path, time_interval_dir, narration_dir, frame_emb_input_dir, time_interval_name, csv_file_name) # get simarlity
#     print(f"model_type_folder: {model_type_folder}, overall_average_cos: {overall_average_cos}, buggy_video: {buggy_video}")
   

# #     ###---------- Demo_random ClipScore ######
#     model_type_folder = "gpt_random_new_0901"

#     time_interval_name = "time_interval_corrected" # code will round to nearest integer
    
#     time_interval_dir = demo_dir / model_type_folder # demo_random
#     frame_folder = demo_dir / model_type_folder / "frames"
#     frame_emb_input_dir = demo_dir / model_type_folder / "embeddings"
#     generate_image_emb(frame_folder, video_name_path, frame_emb_input_dir) # get frame input
#     overall_average_cos, buggy_video = get_overall_similarity_score(video_name_path, time_interval_dir, narration_dir, frame_emb_input_dir, time_interval_name, csv_file_name, "gen") # get simarlity
#     print(f"model_type_folder: {model_type_folder}, overall_average_cos: {overall_average_cos}, buggy_video: {buggy_video}")
   
#     ###---------- Demo_title  ClipScore ######
#     model_type_folder = "gpt_title_new_0901"

#     time_interval_name = "time_interval_corrected" # selected interval, that is the frame number in body contents
    
#     time_interval_dir = demo_dir / model_type_folder # demo_title
#     frame_folder = demo_dir / model_type_folder / "frames"
#     frame_emb_input_dir = demo_dir / model_type_folder / "embeddings"
#     generate_image_emb(frame_folder, video_name_path, frame_emb_input_dir) # get frame input
#     overall_average_cos, buggy_video = get_overall_similarity_score(video_name_path, time_interval_dir, narration_dir, frame_emb_input_dir, time_interval_name, csv_file_name) # get simarlity
#     print(f"model_type_folder: {model_type_folder}, overall_average_cos: {overall_average_cos}, buggy_video: {buggy_video}")
   