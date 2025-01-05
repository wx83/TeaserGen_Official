from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from PIL import Image
import os
import pathlib
import numpy as np
from pathlib import Path
from utils import load_json, load_txt
import torch

def load_json_l(json_file_path):
    # List to store the loaded data
    data = []
    # Open the file and load each line as a separate JSON object
    with open(json_file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

class SentenceImageDataset(Dataset):
    def __init__(self, docu_list, json_file_path, image_emb_dir, sentence_emb_dir, sentence_json, scene_dir=Path("/home/weihanx/videogpt/deepx_data7/scene_intro")):
        """
        Args:
            docu_list: training set
            json_file_path: data annotation on training set
            sentence_emb_dir: sentence embedding directory
            image_emb_dir: image embedding directory
            sentence_json: the text for each video
        """
        self.docu_list = docu_list

        self.sentence_emb_dir = sentence_emb_dir
        self.image_emb_dir = image_emb_dir
        # self.tokenizer = tokenizer
        # self.transform = transform # already processed
        self.annotation = load_json(json_file_path)
        self.sentence_json = sentence_json
        self.scene_dir = scene_dir
    def __len__(self):
        return len(self.docu_list) # number of documnet

    def __getitem__(self, idx):
        # Load and tokenize the sentence
        """
        TODO: sentence should be /home/weihanx/videogpt/workspace/sent_text_emb/_62gPg2Vv6c_0.npy: videoname + sentence number(all selected after remove_: should double check)
        frame: load the whole numpy file and select the frame number
        """
        docu_name = self.docu_list[idx]
        sentence_embeddings = [] # all sentences in this video clip
        image_embeddings = []
        sentence_text_list = []
        # print(f"docuname = {docu_name}")
        # text input
        text_dict = load_json(self.sentence_json) # all text is in this text dict
        for clip_dict in self.annotation[docu_name]: # the doucmentary that I am interested in
            clip_name = clip_dict["video"] # find the video that has different size
            # load the scene annotation for this video
            # scene_path = self.scene_dir / clip_name[0] / f"{clip_name}.npy"
            # scene_annotation = np.load(scene_path, allow_pickle=True)
            # print(f"clip_name = {clip_name}")
            sentence_indices = clip_dict["sentences"] # indicate frame embedding belonging to which sentecnes

            # print(f"len of sentence = {len(sentence_indices)}")
            for sentence_idx in sentence_indices:
                sentence_emb_path = self.sentence_emb_dir / f"{clip_name}.npy" # 1iuQjyMP8_c_0
                # also load text
                sentence_text = text_dict[clip_name] # this video query text
                sentence_embedding = np.load(sentence_emb_path, allow_pickle=True)  # Shape: (1, 768)
                # print(f"video_name = {clip_name}, sentence_embedding: {sentence_embedding.shape}")
                sentence_embeddings.append(sentence_embedding)
                sentence_text_list.append(sentence_text) # number of repetetivion for diffsuion prior
                image_indices = clip_dict["frame_indices"]
                # print(f"image_indices: {image_indices}")
            # apply transformer prior before saving
            if self.image_emb_dir != None:
                image_emb_path = self.image_emb_dir / f"{docu_name}_clip.npy" # all number of frames
                image_embedding = np.load(image_emb_path, allow_pickle=True)
                clip_length = image_embedding.shape[0]
                
                for image_idx in image_indices:
                    if clip_length == image_idx:
                        # last one take upper bound because rounding
                        image_idx = image_idx - 1
                    image_embeddings.append(image_embedding[image_idx].reshape(1, -1))  # Shape: (1, 2048)
        if self.image_emb_dir != None:
            image_embeddings = np.vstack(image_embeddings)  # Shape: (sequence_len, 2048)
        if self.image_emb_dir == None:
            image_embeddings = None
            # Concatenate all sentence embeddings along the sequence dimension
        sentence_embeddings = np.vstack(sentence_embeddings)  # Shape: (sequence_len, 768)
        # print(f"docu_name: {docu_name}, sentence_embeddings shape: {sentence_embeddings.shape}, image_embeddings shape: {image_embeddings.shape}")
        # print(f"sentence_text_list = {sentence_text_list}")
        # print(f"shape of sentence = {sentence_embeddings.shape}")
        
        return sentence_embeddings, image_embeddings, docu_name, sentence_text_list # should return a tuple: sentence embeing, image embeding
   
    @ staticmethod # not need for  any instance of varibale in the class
    def collate_fn(batch):
        # Unpack the batch into separate lists of sentence embeddings and image embeddings
    #    sentence_text_list is also a list, list of list
        sentences, images, docu_name, sentence_text_list = zip(*batch)
        if images is None:
            images = [None] * len(sentences)
        # Calculate the maximum sequence length in the batch
        max_sentence_len = max(s.shape[0] for s in sentences)
        # max_image_len = max(i.shape[0] for i in images)

        # Ensure consistency: max sequence length should be the same for both sentences and images
        # max_seq_len = max(max_sentence_len, max_image_len)
        max_seq_len = max_sentence_len
        # Initialize lists to hold the padded data and masks
        padded_sentences = []
        padded_images = []
        sentence_masks = []
        # padded_scene_annotations = []
        for sentence_emb, image_emb in zip(sentences, images):
            # Get the lengths of the current sentence and image embeddings
            sentence_len = sentence_emb.shape[0]

            # assert len(scene_annotation) == sentence_len.shape[0], f"during training, should have equal length" # This is for training, sentence length should be the same as scene annotation
            # Pad the sentence embeddings to the max sequence length
            sentence_padding_len = max_seq_len - sentence_len
            sentence_padding = np.zeros((sentence_padding_len, sentence_emb.shape[1])) # pad by zero
            padded_sentence = np.vstack([sentence_emb, sentence_padding])
            # shaoe: (seq_len, )
            # padded_scene_annotation = np.concatenate([scene_annotation, np.zeros(sentence_padding_len)])
            # padded_scene_annotations.append(padded_scene_annotation) # padd to desired length for each sentence
            padded_sentences.append(padded_sentence)
            
            # Create the sentence mask: pytorch build in mask: 1 should be ignored, 0 should be kept
            sentence_mask = np.concatenate([np.zeros(sentence_len), np.ones(sentence_padding_len)])
            sentence_masks.append(sentence_mask)
            if image_emb is None:
                padded_image = np.ones((max_seq_len, 768))  # Assuming image embeddings should have 768 dimensions
            else:
                image_len = image_emb.shape[0]
                # Pad the image embeddings to the max sequence length
                image_padding_len = max_seq_len - image_len
                image_padding = np.zeros((image_padding_len, image_emb.shape[1]))
                padded_image = np.vstack([image_emb, image_padding])
            padded_images.append(padded_image)

        # Stack the padded sentences, images, and masks into batched tensors
        # scene_batch = torch.tensor(np.stack(padded_scene_annotations), dtype=torch.float32)  # Shape: (batch_size, max_seq_len)
        sentence_batch = torch.tensor(np.stack(padded_sentences), dtype=torch.float32)  # Shape: (batch_size, max_seq_len, 768)
        image_batch = torch.tensor(np.stack(padded_images), dtype=torch.float32)  # Shape: (batch_size, max_seq_len, 2048)
        # sentnce and image mask should be the same
        sentence_mask_batch = torch.tensor(np.stack(sentence_masks), dtype=torch.float32)  # Shape: (batch_size, max_seq_len)
        # name list
        docu_name = list(docu_name)
        sentence_text_list_list = list(sentence_text_list)
        # image embedding can share the same mask with sentence embedding, as they are the same length, but during loss, need to remove those padded
        return sentence_batch, image_batch, sentence_mask_batch, docu_name, sentence_text_list_list
if __name__ == "__main__":
    image_emb_dir_intro = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_frame_768_emb_intro") # This is the embedding from teaser
    image_emb_dir_main = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_frame_768_emb_main") # This is the embedding from teaser
    sentence_emb_intro_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_text_768_intro")
    sentence_emb_main_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_text_768_main")
    train_annotate_intro = Path("/home/weihanx/videogpt/workspace/transformer_prior/intro/train_0103_intro_comb.json")
    train_annotate_main = Path("/home/weihanx/videogpt/workspace/transformer_prior/main/train_0103_main_comb.json")
    sentence_json_intro = Path("/home/weihanx/videogpt/workspace/transformer_prior/intro/intro_text_name.json") # why sentence dir
    sentence_json_main = Path("/home/weihanx/videogpt/workspace/transformer_prior/main/main_text_name.json")

    # combined_dict_main = load_json(train_annotate_main) # smaller set
    # train_eval_list_main = list(combined_dict_main.keys()) 
    combined_dict_intro = load_json(train_annotate_intro) # smaller set
    train_eval_list_intro = list(combined_dict_intro.keys())
    # common_keys = list(set(train_eval_list_main).intersection(train_eval_list_intro))
    common_keys =train_eval_list_intro
    print(f"len of common keys = {len(common_keys)}")
    train_list = common_keys
    # train_list = common_keys[:int(len(common_keys)*0.8)]
    # eval_list = common_keys[int(len(common_keys)*0.8):]
    # can be combined or trained separately but # use the same set of filenames
    train_dataset_intro = SentenceImageDataset(train_list, train_annotate_intro, image_emb_dir_intro, sentence_emb_intro_dir, sentence_json_intro)
    # train_dataset_main = SentenceImageDataset(train_list, train_annotate_main, image_emb_dir_main, sentence_emb_main_dir, sentence_json_main)
    # concatenate two dataset
    train_dataset = train_dataset_intro
    # train_dataset = torch.utils.data.ConcatDataset([train_dataset_intro, train_dataset_main]) # should learn separately or together? They should not have temporal dependence
    bsz = 32
    # val_dataset_intro = SentenceImageDataset(eval_list, train_annotate_intro, image_emb_dir_intro, sentence_emb_intro_dir, sentence_json_intro)
    # val_dataset_main = SentenceImageDataset(eval_list, train_annotate_main, image_emb_dir_main, sentence_emb_main_dir, sentence_json_main)
    # val_dataset = torch.utils.data.ConcatDataset([val_dataset_intro, val_dataset_main])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bsz,  # Set your desired batch size
        shuffle=True,  # Whether to shuffle the data at each epoch
        collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
    )
    print(f"len of train_loader = {len(train_dataloader)}") # final check in dataloader

    # valid_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=bsz,  # Set your desired batch size
    #     shuffle=True,  # Whether to shuffle the data at each epoch
    #     collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
    # )
    for sentence_batch, image_batch, sentence_mask, docu_name, sentence_text in train_dataloader:
        print(f"Loader Sentence batch shape: {sentence_batch.shape}")  # Should be (batch_size, 768)
        print(f"Loader Image batch shape: {image_batch.shape}")  # already check: 4, 83, 512
        print(f"Loader Sentence mask shape: {sentence_mask.shape}")
        print(f"Loader docu_name: {docu_name}")
        print(f"Loader list = {len(sentence_text[0])},{len(sentence_text)}")
        