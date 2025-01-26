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
    pass