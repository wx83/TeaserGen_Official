import clip
import numpy as np
import torch
import re
from pathlib import Path
# Load the CLIP model and tokenizer
from utils import load_txt, save_txt
import pandas as pd
import json
import random
import shutil
import torch.nn as nn
# import nltk
# from nltk.tokenize import sent_tokenize
import torch
import sys

random.seed(42) # 768 for vit-l/14
# https://github.com/openai/CLIP/issues/111 potential for multiple gpu

import re

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
def process_text_with_clip(text, model, diffusion_prior=False):
    """Get the CLIP embedding for a given text."""
    max_context_length = 77  # CLIP's token limit

    # Truncate input text to 77 words
    words = text.split()  # Split text into words
    if len(words) > max_context_length:
        print(f"Input text too long ({len(words)} words): Truncating to {max_context_length} words.")
        text = " ".join(words[:max_context_length])  # Truncate at word level

    # Tokenize text and move to the appropriate device
    try:
        text_inputs = clip.tokenize([text]).to("cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        # further truncate the text
        text = " ".join(words[:max_context_length//2])  # Truncate at word level
        text_inputs = clip.tokenize([text]).to("cuda" if torch.cuda.is_available() else "cpu")
    if diffusion_prior:
        # Process with diffusion prior
        image_embeds = diffusion_prior.sample(text_inputs)
        print(f"image_embeds shape: {image_embeds.shape}")
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        print(f"text_features shape: {text_features.shape}")
        text_features = image_embeds + text_features
    else:
        # Process text embeddings without diffusion prior
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)

    return text_features.cpu().numpy()



# Loading data
def generate_text_emb_demo():
    video_jsonl = Path("/home/weihanx/videogpt/deepx_data6/dataset/annotation_0904.jsonl")

    with open(video_jsonl, 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    with open(video_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]

    qid_list = []

    for pair in data:
        for key, value in pair.items():
            # try:
            qid = key
            query = value['query']
            
            # Check if the text is too long for CLIP's tokenization limit
            if check_if_text_too_long(query, model, max_token_length=77):
                # If too long, split by punctuation
                
                sentences = split_text_by_punctuation(query)
                
                # Get embeddings for each sentence and concatenate
                text_features_list = []
                for sentence in sentences:
                    # print(sentence)
                    # add diffusioin prior for each sentences
                    text_features = process_text_with_clip(sentence, model)


                    text_features_list.append(text_features) # already numpy
                # except Exception as e:
                #     sentences = split_text_into_sentences(query)
                #     # Get embeddings for each sentence and concatenate
                #     text_features_list = []
                #     for sentence in sentences:
                #         print(sentence)
                #         text_features = process_text_with_clip(sentence, model)
                #         text_features_list.append(text_features)
                # Concatenate embeddings
                text_features_concat = np.concatenate(text_features_list, axis=0)
                
                # Apply the WeightedPool layer to reduce the result to (1, 768)
                # text_features_concat_tensor = torch.tensor(text_features_concat).to("cuda")  # Convert to tensor
                # text_features_concat_tensor = text_features_concat_tensor.unsqueeze(0)  # Add batch dimension, only for weighted
                print(f"before average {text_features_concat.shape}") # sequence legnth, 768
                with torch.no_grad():
                    # pooled_output = weighted_pool(text_features_concat_tensor).cpu().numpy()
                    pooled_output = np.mean(text_features_concat, axis=0, keepdims=True) # average over sequence dimension
                    print(f"pooled_output shape: {pooled_output.shape}") # already 1, 768
                # pooled_output = pooled_output.squeeze(0) # remove batch dimesion, 1, 768
                # # add diffusioin prior
                # valid_range = (pooled_output >= 0) & (pooled_output < 49408)

                # # If any value is out of range, valid_range will contain a `False`
                # if not valid_range.all():
                #     print("Warning: Some values in pooled_output are out of the range [0, 49407]")
                #     # Optionally, print the indices and values that are out of range
                #     out_of_range_indices = torch.nonzero(~valid_range)
                #     print(f"qid = {qid}")
                #     print("Out of range values:", pooled_output[out_of_range_indices])
                # else:
                    
                #     print("All values are within the range [0, 49407]")
            else:
                # If not too long, process the entire text with CLIP directly
                pooled_output = process_text_with_clip(query, model)

            # Save the pooled or concatenated embeddings as an .npy file
            assert pooled_output.shape == (1, 512), f"Unexpected shape: {pooled_output.shape}"
            print(f"Saving {qid}...{pooled_output.shape}")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{qid}.npy"
            np.save(output_path, pooled_output)
                
            

            # except Exception as e:
            #     print(f"Error processing {qid}: {e}")
            #     qid_list.append(qid)
            #     continue
    # save_txt(qid_list, )
    save_txt(output_dir / "error_qids.txt", qid_list)


def generate_text_emb_train(video_name_path, input_dir, output_dir, model):
    video_names = load_txt(video_name_path)

    # /home/weihanx/videogpt/deepx_data6/screen_play_main
    for video in video_names:
        output_path = output_dir / f"{video}_0.npy"
        if output_path.exists() or video == "Kap1dQnkKPw" or video == "acMYHAmNLlE":
            print(f"Already processed video: {video}")
            continue
        try:
            csv_path = input_dir / video[0] / video / "script_timeline.csv"
            df = pd.read_csv(csv_path) # ensure the row number is not altered 
            index = 0 # for each video, query start with 0
            print(f"video = {video}")
            for row, data in df.iterrows():
                text = data['Text']
                # if only want to have narration, just specify
                # speaker = data['speaker']: SPEAKER_00, find those row number with certain speaker index
                # print(f"text = {text}") # not in quote
                if not text.isascii(): # exclude those non-english
                    continue
                if check_if_text_too_long(text, model, max_token_length=77):

                    sentences = split_text_by_punctuation(text)
                    
                    text_features_list = []
                    for sentence in sentences:
                        text_features = process_text_with_clip(sentence, model)


                        text_features_list.append(text_features) # already numpy

                    text_features_concat = np.concatenate(text_features_list, axis=0)
                    

                    print(f"before average {text_features_concat.shape}") # sequence legnth, 768
                    with torch.no_grad():
                        # pooled_output = weighted_pool(text_features_concat_tensor).cpu().numpy()
                        pooled_output = np.mean(text_features_concat, axis=0, keepdims=True) # average over sequence dimension
                        print(f"pooled_output shape: {pooled_output.shape}") # already 1, 768
                else:
                    # If not too long, process the entire text with CLIP directly
                    pooled_output = process_text_with_clip(text, model)
                # Save the pooled or concatenated embeddings as an .npy file
                assert pooled_output.shape == (1, 768), f"Unexpected shape: {pooled_output.shape}"
                output_path = output_dir / f"{video}_{row}.npy"
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_path, pooled_output)
                index += 1
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue

def generate_text_emb_gpt(video_name_path, input_dir, output_dir):
    video_names = load_txt(video_name_path)
    for video in video_names:
        csv_path = input_dir / video[0] / video / "story_script.csv"
        df = pd.read_csv(csv_path)
        for row, data in df.iterrows():
            id = data['id']
            text = data['text']
            if check_if_text_too_long(text, model, max_token_length=77):
                # If too long, split by punctuation
                
                sentences = split_text_by_punctuation(text)
                
                # Get embeddings for each sentence and concatenate
                text_features_list = []
                for sentence in sentences:
                    # print(sentence)
                    # add diffusioin prior for each sentences
                    text_features = process_text_with_clip(sentence, model)


                    text_features_list.append(text_features) # already numpy
                # except Exception as e:
                #     sentences = split_text_into_sentences(query)
                #     # Get embeddings for each sentence and concatenate
                #     text_features_list = []
                #     for sentence in sentences:
                #         print(sentence)
                #         text_features = process_text_with_clip(sentence, model)
                #         text_features_list.append(text_features)
                # Concatenate embeddings
                text_features_concat = np.concatenate(text_features_list, axis=0)
                
                # Apply the WeightedPool layer to reduce the result to (1, 768)
                # text_features_concat_tensor = torch.tensor(text_features_concat).to("cuda")  # Convert to tensor
                # text_features_concat_tensor = text_features_concat_tensor.unsqueeze(0)  # Add batch dimension, only for weighted
                print(f"before average {text_features_concat.shape}") # sequence legnth, 768
                with torch.no_grad():
                    # pooled_output = weighted_pool(text_features_concat_tensor).cpu().numpy()
                    pooled_output = np.mean(text_features_concat, axis=0, keepdims=True) # average over sequence dimension
                    print(f"pooled_output shape: {pooled_output.shape}") # already 1, 768
            else:
                # If not too long, process the entire text with CLIP directly
                pooled_output = process_text_with_clip(text, model)
            qid = f"{video}_{id}"
            # Save the pooled or concatenated embeddings as an .npy file
            assert pooled_output.shape == (1, 512), f"Unexpected shape: {pooled_output.shape}"
            print(f"Saving {qid}...{pooled_output.shape}")
            
            output_path = output_dir / f"{qid}.npy"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_path, pooled_output)

def generate_text_emb_a2summ(video_name_path, input_dir, output_dir):
    video_names = load_txt(video_name_path)
    for video in video_names:
        for i in range(10):
            txt_path = input_dir / f"{video}_{i}.txt"
            text = load_txt(txt_path)[0]
            
            if check_if_text_too_long(text, model, max_token_length=77):
                # If too long, split by punctuation
                
                sentences = split_text_by_punctuation(text)
                
                # Get embeddings for each sentence and concatenate
                text_features_list = []
                for sentence in sentences:
                    # print(sentence)
                    # add diffusioin prior for each sentences
                    text_features = process_text_with_clip(sentence, model)


                    text_features_list.append(text_features) # already numpy
                # except Exception as e:
                #     sentences = split_text_into_sentences(query)
                #     # Get embeddings for each sentence and concatenate
                #     text_features_list = []
                #     for sentence in sentences:
                #         print(sentence)
                #         text_features = process_text_with_clip(sentence, model)
                #         text_features_list.append(text_features)
                # Concatenate embeddings
                text_features_concat = np.concatenate(text_features_list, axis=0)
                
                # Apply the WeightedPool layer to reduce the result to (1, 768)
                # text_features_concat_tensor = torch.tensor(text_features_concat).to("cuda")  # Convert to tensor
                # text_features_concat_tensor = text_features_concat_tensor.unsqueeze(0)  # Add batch dimension, only for weighted
                print(f"before average {text_features_concat.shape}") # sequence legnth, 768
                with torch.no_grad():
                    # pooled_output = weighted_pool(text_features_concat_tensor).cpu().numpy()
                    pooled_output = np.mean(text_features_concat, axis=0, keepdims=True) # average over sequence dimension
                    print(f"pooled_output shape: {pooled_output.shape}") # already 1, 768
            else:
                # If not too long, process the entire text with CLIP directly
                pooled_output = process_text_with_clip(text, model)
            # Save the pooled or concatenated embeddings as an .npy file
            assert pooled_output.shape == (1, 512), f"Unexpected shape: {pooled_output.shape}"
            output_path = output_dir / f"{video}_{i}.npy"
            output_dir.mkdir(parents=True, exist_ok=True)
            np.save(output_path, pooled_output)

def save_narr_text_intro(output_dir):
    video_jsonl = Path("/home/weihanx/videogpt/deepx_data6/dataset/annotation_0904.jsonl")

    with open(video_jsonl, 'r') as f:
        data = []
        for line in f:
            data.append(json.loads(line))
    with open(video_jsonl, 'r') as f:
        data = [json.loads(line) for line in f]

    qid_list = []

    for pair in data:
        for key, value in pair.items():
            # try:
            qid = key
            query = value['query']
            
            output_path = output_dir / f"{qid}.txt"
            with open(output_path, 'w') as file:
                file.write(query)
            
            text = load_txt(output_path)
            print(f"text = {text}") # ['txt']

def save_narr_gpt_intro(video_name_path, input_dir, output_dir):
    
    for video in video_name_path:
        csv_path = input_dir / video[0] / video / "story_script.csv"
        df = pd.read_csv(csv_path)
        for row, data in df.iterrows():
            id = data['id']
            text = data['text']
            query_id = f"{video}_{id}.txt"
            file_path = output_dir / query_id
            with open(file_path, 'w') as file:
                file.write(data['text'])


def save_narr_main(video_name_path, input_dir, output_dir):
    """
    Saves extracted text data from CSVs to a JSON file for further processing.
    """
    data = {}  # Initialize the data dictionary to collect all text entries
    video_names = load_txt(video_name_path)  # This function should return a list of video names
    for video in video_names:
        csv_path = Path(input_dir) / video[0] / video / "script_timeline.csv"  # Construct path using pathlib
        if not csv_path.exists():
            print(video)
            continue
        df = pd.read_csv(csv_path)

        for row, row_data in df.iterrows():  # renamed `data` to `row_data` to avoid shadowing
            text = row_data['Text']
            query_id = f"{video}_{row}"  # Constructing a unique identifier
            data[query_id] = text  # Save text into the dictionary using the unique identifier

    # Path to output JSON file
    output_path = Path(output_dir) / "intro_text_name.json"
    with open(output_path, 'w') as f:
        json.dump(data, f)  # Dump the dictionary directly without converting to_dict()


if __name__ == "__main__":
    model, preprocess = clip.load("ViT-L/14", device="cuda") # for evaluation
    # video_name_path = Path("/home/weihanx/videogpt/whisperX/good_file.txt")
    video_name_path = Path("/home/weihanx/videogpt/workspace/transformer_prior/train_val_files_intro.txt")
    # part_1 = Path("/home/weihanx/videogpt/processed.txt")
    # error_file_path = Path("/home/weihanx/videogpt/whisperX/error_file.txt")
    input_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_intro_train")
    annotate_output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/intro")
    embed_output_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_text_768_intro")
    annotate_output_dir.mkdir(parents=True, exist_ok=True)  # Add exist_ok=True to avoid errors if the directory exists
    generate_text_emb_train(video_name_path, input_dir, embed_output_dir, model)
    save_narr_main(video_name_path, input_dir, annotate_output_dir) # to large 

    # part_2 = Path("/home/weihanx/videogpt/processed_2.txt")
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/screen_play_main_2")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/clip_text_768_main")
    # generate_text_emb_train(video_name_path, input_dir, output_dir, part_2, model)
    # video_name_path = Path("/home/weihanx/videogpt/workspace/start_code/eval/final_eval.txt")
    # # video_name_path = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_name.txt")
    # text_save_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/sentence")
    # # output_dir = Path("/home/weihanx/videogpt/deepx_data7/clip_sentence_emb_512_average_gpt")
    # # generate_text_emb_gpt(video_name_path, input_dir, output_dir)
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/testset_a2summ/sentence_text_embed")
    # output_dir.mkdir(parents=True, exist_ok=True)  # Add exist_ok=True to avoid errors if the directory exists
    # # save_narr_text(output_dir)
    # generate_text_emb_a2summ(video_name_path, text_save_dir, output_dir)
    # # # # generate_text_emb_demo()
    # video_name_path = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_name.txt")
    # input_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_narration")
    # # output_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_narr_text")
    # output_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_clip_sentence_emb_512_average_gpt")
    # output_dir.mkdir(parents=True, exist_ok=True)  # Add exist_ok=True to avoid errors if the directory exists
    # generate_text_emb_gpt(video_name_path, input_dir, output_dir)
    # save_narr_gpt(video_name_path, input_dir, output_dir)
    # generate_text_emb_gpt(video_name_path, input_dir, output_dir)