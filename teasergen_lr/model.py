# SSIM Loss
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torchvision import models
from dataset import SentenceImageDataset
import pathlib 
import re
from pathlib import Path
from utils import load_json, save_json, save_txt, load_txt
import sys
sys.path.append('/home/weihanx/videogpt/workspace/transformer_prior')
import clip
import time
import torch.nn.functional as F

from torch.utils.data import DataLoader
import dalle2_pytorch
# from SSIM_Loss import ssim_loss
from dalle2_pytorch import train_configs, OpenAIClipAdapter, DiffusionPriorNetwork, DiffusionPriorTrainer,DiffusionPrior
# Absolute position encoding
# VGG-Based Perceptual Loss
import sys
sys.path.append('/home/weihanx/videogpt/workspace/transformer_prior')
from vgg_loss import vgg_loss
import torch
from torch import optim
from torchvision import io as tio
from torchvision.transforms import functional as TF

class PositionalEncoding(nn.Module): # absolute positional encoding 
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model) # position
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure x is a tensor of shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x



class CustomModel(nn.Module):
    def __init__(self, apply_diffusion_prior, hidden_dim=768, nhead=12, num_layers=6, device="cuda:0"):
        super(CustomModel, self).__init__()
        
        self.position_encoding = PositionalEncoding(d_model=hidden_dim)
        # self.res50_feat = 2048
        # Transformer Encoder
        self.apply_diffusion_prior  = apply_diffusion_prior
        # generate 
        self.scene_embedding = nn.Parameter(torch.zeros(1, 200, hidden_dim)) # extra dimension for group embedding, maximum 200 groups
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)

        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        config_file = "/home/weihanx/videogpt/workspace/transformer_prior/prior_config.json"
        ckpt_file = "/home/weihanx/videogpt/workspace/transformer_prior/best.pth"
        self.device = device
        if self.apply_diffusion_prior ==True:
            prior_network = DiffusionPriorNetwork(
                dim=768,
                depth=12,
                dim_head=64,
                heads=12,
                normformer=True,
                attn_dropout=0,
                ff_dropout=0,
                num_time_embeds=1,
                num_image_embeds=1,
                num_text_embeds=1,
                num_timesteps=1000,
                ff_mult=4,
                max_text_len = 77,
            )
            diffusion_prior = DiffusionPrior(
                net=prior_network,
                clip=OpenAIClipAdapter("ViT-L/14"),
                image_embed_dim=768,
                timesteps=1000,
                cond_drop_prob=0,
                loss_type="l2",
                condition_on_text_encodings=True,

            )

            self.diffusion_prior = diffusion_prior.to(device)
            loaded_obj = torch.load(ckpt_file, map_location="cuda:0")
            diffusion_prior.load_state_dict(loaded_obj, strict=True)

    def average_embedding(self, predicted_embeddings, map_mean_list):
        unique_indices = set(map_mean_list)
        averaged_embeddings = []

        for idx in unique_indices:
            # Find the indices in the embeddings tensor that match the current group index
            mask = [i for i, x in enumerate(map_mean_list) if x == idx]
            
            # Select embeddings for the current group and compute the mean
            group_embeddings = predicted_embeddings[mask]
            averaged_embedding = group_embeddings.mean(dim=0)
            
            # Add to the list of averaged embeddings
            averaged_embeddings.append(averaged_embedding)
        result = torch.stack(averaged_embeddings)
        return result
    
    def truncate_sentence(self, sentence):
        # Truncate the sentence to half of its current length
        return sentence[:len(sentence) // 2]

    def check_sub_sentences(self, sub_sentences):
        sentence_check = []
        for sentence in sub_sentences:
            checked_sentence = self.try_tokenize(sentence) # just for check
            print(f"final saved checked sentence = {len(checked_sentence)}")
            sentence_check.append(checked_sentence)

        return sentence_check

    def try_tokenize(self, sentence):
        try:
            # Attempt to tokenize the sentence
            tokenized_texts = clip.tokenize(sentence, context_length=77).to(self.device)
            return sentence
        except Exception as e:
            print(f"further split is needed for sentence = {sentence}")
            # If an exception occurs, truncate the sentence and try again
            truncated_sentence = self.truncate_sentence(sentence)
            return self.try_tokenize(truncated_sentence) # recursive until it is procesable, only truncate, not split to two

    def get_dp(self, sentences, device="cuda:0"):
        map_mean_list = []
        all_sentences = []

        # print(f"num of sentences in this sample = {len(sentences)}")
        for idx, sub_sent in enumerate(sentences):
            try:
                tokenized_texts = clip.tokenize(sub_sent, context_length=77).to(device)
                map_mean_list.append(idx)
                all_sentences.append(sub_sent)
            except Exception as e:
                # print(f"not able to process long sentence")
                sub_sentences = re.split(r'[，。！？,.;:]', sub_sent)
                sub_sentences = [s.strip() for s in sub_sentences if s.strip()]
                sub_sentences = self.check_sub_sentences(sub_sentences)
                # need check if it is procesable not
                all_sentences.extend(sub_sentences)
                len_of_sub = len(sub_sentences)
                my_list = [idx] * len_of_sub
                map_mean_list.extend(my_list)

        # Now process the sentences in batches
        batch_size = 512
        all_predicted_embeddings = []
        for i in range(0, len(all_sentences), batch_size):
            batch_sentences = all_sentences[i:i + batch_size]
            tokenized_texts = clip.tokenize(batch_sentences, context_length=77).to(device)
            predicted_embeddings = self.diffusion_prior.sample(tokenized_texts, timesteps=64, cond_scale=1.0)  # Shape: (batch_size, 768)
            all_predicted_embeddings.append(predicted_embeddings)

        # Concatenate all the predicted embeddings from each batch
        all_predicted_embeddings = torch.cat(all_predicted_embeddings, dim=0)  # Shape: (num_all_sentences, 768)
        print(f"len of all sentences = {len(all_sentences)}, {all_predicted_embeddings.shape}")
        # Create average embeddings for those belonging to the same sentence
        update_embedding = self.average_embedding(all_predicted_embeddings, map_mean_list)
        print(f"update_embedding shape = {update_embedding.shape}")

        torch.cuda.empty_cache()
        return update_embedding  # Shape: (num_sentences, 768)


    def forward(self, sentence_embeddings, mask, sentence_text):

        if self.apply_diffusion_prior == True:
            for b in range(len(sentence_text)):
                # Batch process sentences in sentence_text[b]
                predicted_embedding_list = self.get_dp(sentence_text[b], self.device)  # Now handles batch input
                # No need to use a loop; predicted_embedding_list is already in the form (num_sentences, 768)
                total_len_dp = predicted_embedding_list.shape[0]

                sentence_embeddings[b][:total_len_dp] += predicted_embedding_list
        # print(f"sentence embedidng into transfomre = {sentence_embeddings.shape}")
        sentence_embeddings = self.position_encoding(sentence_embeddings)

        transformer_output = self.transformer_encoder(sentence_embeddings, src_key_padding_mask=mask)
        # add trainable group embedding
        # output_embedding = transformer_output + self.scene_embedding # add a group embeeding
        
        return transformer_output