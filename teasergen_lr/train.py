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
from decoding import *
import clip
import time
import torch.nn.functional as F

from torch.utils.data import DataLoader
from inference import *
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

from model import CustomModel

def construct_pairs_optimized(batch_embeddings, batch_labels):
    """
    Constructs positive and negative pairs within a batch using optimized tensor operations.
    """
    batch_size, seq_length, emb_dim = batch_embeddings.size()
    
    # Create all pair indices
    triu_indices = torch.triu_indices(seq_length, seq_length, offset=1)
    j_indices = triu_indices[0]
    k_indices = triu_indices[1]

    # Expand labels to compare all pairs
    expanded_labels = batch_labels.unsqueeze(2).expand(-1, -1, seq_length)
    label_j = expanded_labels.gather(2, j_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1))
    label_k = expanded_labels.gather(2, k_indices.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1))

    # Determine which pairs are positive and which are negative
    positives_mask = (label_j == label_k).squeeze(2)
    negatives_mask = (label_j != label_k).squeeze(2)

    # Gather embeddings according to these masks
    emb_expanded = batch_embeddings.unsqueeze(1).expand(-1, seq_length, -1, -1)
    emb_j = emb_expanded.gather(2, j_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, emb_dim))
    emb_k = emb_expanded.gather(2, k_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, emb_dim))

    positive_pairs = torch.stack((emb_j[positives_mask], emb_k[positives_mask]), dim=1)
    negative_pairs = torch.stack((emb_j[negatives_mask], emb_k[negatives_mask]), dim=1)

    return positive_pairs, negative_pairs

def custom_contrastive_loss(positive_pairs, negative_pairs, margin=1.0):
    """
    Computes a contrastive loss where the loss is minimized for positive pairs
    and maximized for negative pairs.

    Args:
    - positive_pairs (Tensor): Tensor of shape (num_pos_pairs, 2, emb_dim)
    - negative_pairs (Tensor): Tensor of shape (num_neg_pairs, 2, emb_dim)
    - margin (float): Margin by which negative pairs should be separated.

    Returns:
    - loss (Tensor): The computed contrastive loss.
    """
    # Calculate cosine similarity for positive and negative pairs
    pos_sim = torch.cosine_similarity(positive_pairs[:, 0, :], positive_pairs[:, 1, :])
    neg_sim = torch.cosine_similarity(negative_pairs[:, 0, :], negative_pairs[:, 1, :])

    # Calculate the mean of similarities
    mean_pos_sim = pos_sim.mean()
    mean_neg_sim = neg_sim.mean()

    # Loss for positive pairs: we want to maximize the mean_pos_sim (i.e., minimize (1 - mean_pos_sim))
    positive_loss = 1 - mean_pos_sim


    negative_loss = F.relu(margin - mean_neg_sim)

    # Total loss is the sum of positive loss and negative loss
    total_loss = positive_loss + negative_loss

    return total_loss

def check_similarity(matrix1, matrix2):
    # matrix1: batch size, seq len, 768
    # matrix2: batch size, seq len, 768
    bsz, seq_len, dim = matrix1.shape
    for i in range(bsz):
        sample_wise_sim = []
        for j in range(seq_len):
            # shape matrxi1[i][j] = 768
            l2_sim = F.pairwise_distance(matrix1[i][j].unsqueeze(0), matrix2[i][j].unsqueeze(0), p=2)
            sample_wise_sim.append(l2_sim.squeeze().item())
        print(f"bsz = {i}, average = {sample_wise_sim}")

def check_pairwise_similarity(matrix1):
    bsz, seq_len, dim = matrix1.shape
    all_distances = []
    
    # Iterate over each sample in the batch
    for i in range(bsz):
        # Get the current sample and reshape to [seq_len, 1, dim] to broadcast
        current_sample = matrix1[i].unsqueeze(1)  # Shape: [seq_len, 1, dim]

        distances = F.pairwise_distance(
            current_sample.expand(seq_len, seq_len, dim),
            current_sample.transpose(0, 1).expand(seq_len, seq_len, dim),
            p=2
        )
        
        # Store the upper triangle of the distance matrix, since it is symmetric
        distances = distances.triu(1)  # Keep upper triangle, setting lower triangle to zero
        all_distances.append(distances)
    
    # Convert list of distance matrices to a batched tensor
    all_distances = torch.stack(all_distances)  # Shape: [bsz, seq_len, seq_len]
    print(f"all distances = {all_distances}")
    return all_distances

def setup_paths(narrator_only):
    paths = {}
    base_path = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding")
    workspace_path = Path("/home/weihanx/videogpt/workspace/transformer_prior")
    if narrator_only:
        paths['sentence_emb_intro'] = base_path / "clip_text_768_intro_narrator"
        paths['sentence_emb_main'] = base_path / "clip_text_768_main_narrator"
        paths['train_annotate_intro'] = workspace_path / "intro/train_0103_intro_comb_narrator.json"
        paths['train_annotate_main'] = workspace_path / "main/train_0103_main_comb_narrator.json"
    else:
        paths['sentence_emb_intro'] = base_path / "clip_text_768_intro"
        paths['sentence_emb_main'] = base_path / "clip_text_768_main"
        paths['train_annotate_intro'] = workspace_path / "intro/train_0103_intro_comb.json"
        paths['train_annotate_main'] = workspace_path / "main/train_0103_main_comb.json"
    paths['image_emb_intro'] = base_path / "clip_frame_768_emb_intro"
    paths['image_emb_main'] = base_path / "clip_frame_768_emb_main"
    paths['sentence_json_intro'] = workspace_path / "intro/intro_text_name.json"
    paths['sentence_json_main'] = workspace_path / "main/main_text_name.json"

    return paths

def data_split(paths):
    combined_dict_main = load_json(paths['sentence_json_main']) # smaller set
    train_eval_list_main = list(combined_dict_main.keys()) 
    combined_dict_intro = load_json(paths['sentence_json_intro']) # smaller set
    train_eval_list_intro = list(combined_dict_intro.keys())
    common_keys = list(set(train_eval_list_main).intersection(train_eval_list_intro))
    print(f"len of common keys = {len(common_keys)}")
    train_list = common_keys[:int(len(common_keys)*0.8)]
    eval_list = common_keys[int(len(common_keys)*0.8):]
    return train_list, eval_list

def create_datasets(paths, train_list, eval_list):

    datasets = {
        'train_intro': SentenceImageDataset(train_list, paths['train_annotate_intro'], paths['image_emb_intro'], paths['sentence_emb_intro'], paths['sentence_json_intro']),
        'train_main': SentenceImageDataset(train_list, paths['train_annotate_main'], paths['image_emb_main'], paths['sentence_emb_main'], paths['sentence_json_main']),
        'eval_intro': SentenceImageDataset(eval_list, paths['train_annotate_intro'], paths['image_emb_intro'], paths['sentence_emb_intro'], paths['sentence_json_intro']),
        'eval_main': SentenceImageDataset(eval_list, paths['train_annotate_main'], paths['image_emb_main'], paths['sentence_emb_main'], paths['sentence_json_main'])
    }
    datasets['train'] = torch.utils.data.ConcatDataset([datasets['train_intro'], datasets['train_main']])
    datasets['eval'] = torch.utils.data.ConcatDataset([datasets['eval_intro'], datasets['eval_main']])
    return datasets

def setup_dataloader(datasets, batch_size):
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=SentenceImageDataset.collate_fn),
        'valid': DataLoader(datasets['eval'], batch_size=batch_size, shuffle=True, collate_fn=SentenceImageDataset.collate_fn)
    }


def train(apply_diffusion_prior, type_loss, device, narrator_only, output_dir): 
    paths = setup_paths(narrator_only)
    
    train_list, eval_list = data_split(paths)

    datasets = create_datasets(paths, train_list, eval_list)
    train_dataloader = setup_dataloader(datasets, 8)['train']
    valid_dataloader = setup_dataloader(datasets, 8)['valid']

    model = CustomModel(apply_diffusion_prior, device=device)

    model = model.to(device)

    if type_loss == "l2":
        criterion = nn.MSELoss()
    elif type_loss == "vgg_perceptual":
        criterion = vgg_loss.WeightedLoss([vgg_loss.VGGLoss(shift=2),
                                    nn.MSELoss(),
                                    vgg_loss.TVLoss(p=1)],
                                    [1, 40, 10]).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    num_epochs = 50

    # # Training loop
    # from time import time
    start_time = time.time()
    lowest_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch in train_dataloader:
            # print(f"Processing batch")
            sentences, images, mask, docu_name, sentence_text = batch

            print(f"sentences shape: {sentences.shape}, images shape: {images.shape}, mask shape{mask.shape}")
            sentences = sentences.to(device)
            images = images.to(device)
            mask = mask.to(device)
            # print(f"sentences shape: {sentences.shape}, images shape: {images.shape}")
            # Forward pass
            
            transformer_output = model(sentences, mask, sentence_text) # batch size, seq_len, 768
            mask_expanded = mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
            masked_transformer_output = transformer_output * mask_expanded
            # positive_pairs, negative_pairs = construct_pairs_optimized(masked_transformer_output, scene_batch)
            # contrastive_loss = custom_contrastive_loss(positive_pairs, negative_pairs, margin=1.0)
            transformer_output_sum = masked_transformer_output.sum(dim=1)
            non_padded_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
            transformer_output_mean = transformer_output_sum / non_padded_count

            masked_images = images * mask_expanded  

            image_embeddings_sum = masked_images.sum(dim=1)
            image_non_padded_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            image_embeddings_mean = image_embeddings_sum / image_non_padded_count
            # average mean: batch size, dimension
            # print(f"transformer_output_mean shape = {transformer_output_mean.shape}, image_embeddings_mean shape = {image_embeddings_mean.shape}")
            loss = criterion(transformer_output_mean, image_embeddings_mean)
            loss = loss  # fix thed bug 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate the loss for this epoch
            epoch_loss += loss.item()
        # Print loss for the epoch
        check_similarity(masked_transformer_output, masked_images)
        # check simialrity of output
        print(f"checking output pairwise similarity")
        check_pairwise_similarity(masked_transformer_output)
        print(f"checking ground truth image pairwise similarity")
        check_pairwise_similarity(masked_images)
        print(f"Epoch [{epoch+1}/{num_epochs}], train Loss: {epoch_loss/len(train_dataloader):.4f}")

        model.eval()
        val_loss = 0
        for batch in valid_dataloader:
            sentences, images, mask, docu_name, sentence_text = batch

            sentences = sentences.to(device)
            images = images.to(device)
            mask = mask.to(device)

            transformer_output = model(sentences, mask, sentence_text)
            mask_expanded = mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
            masked_transformer_output = transformer_output * mask_expanded

            transformer_output_sum = masked_transformer_output.sum(dim=1)
            non_padded_count = mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
            transformer_output_mean = transformer_output_sum / non_padded_count

            masked_images = images * mask_expanded  

            image_embeddings_sum = masked_images.sum(dim=1)
            image_non_padded_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            image_embeddings_mean = image_embeddings_sum / image_non_padded_count

            loss = criterion(transformer_output_mean, image_embeddings_mean)
            val_loss += loss.item()
        val_loss_normalized = val_loss / len(valid_dataloader)

        # Print validation loss and check for improvement
        print(f"Epoch [{epoch+1}/{num_epochs}], val Loss: {val_loss_normalized:.4f}, lowest_val_loss = {lowest_val_loss}")
        if val_loss_normalized < lowest_val_loss:
            lowest_val_loss = val_loss_normalized
            print(f"save model with lowest val loss = {lowest_val_loss:.4f}")
            output_path = output_dir / f"dp_{apply_diffusion_prior}" /f"loss_{type_loss}"/ f"narrator_only_{narrator_only}" / f"checkpoint_epoch_{epoch}.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
        else:
            break
        # if epoch % 1 == 0:
        #     torch.save(model.state_dict(), f"/home/weihanx/videogpt/workspace/transformer_prior/model_{apply_diffusion_prior}_prior_0926.pth")
    end_time = time.time()
    print(f"training one epoch need = {end_time - start_time}")
    print("Training complete.")




if __name__ == "__main__":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loss  "ssim":
    # train(apply_diffusion_prior=False, type_loss="vgg_perceptual")
    # /home/weihanx/videogpt/workspace/transformer_prior/model_False_1_l2_prior_0103.pth
    body_contents_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/test_embedding/clip_frame_768_emb_main")
    narrator_only = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model_path = Path("/home/weihanx/videogpt/workspace/transformer_prior/model_False_0_l2_prior_0111_narrator_True.pth")
    train(apply_diffusion_prior=True, type_loss="l2", device = device, narrator_only = narrator_only) # retrain by fix bug
   
    # test_phase("gpt", "greedy", True, False, 768) # consider time interval
    # test_phase("original", "greedy", True, True, 768) # consider time interval
    # test_phase("gpt", "greedy", True, True, 768) # running
    # test_phase("original", "beam_search_window", True, False, 768) # running
    # test_phase("original", "greedy", True, True, 768)
    # test_phase("gpt", "beam_search_window", True, True, 768)
    # model_path = Path("/home/weihanx/videogpt/workspace/transformer_prior/model_False_prior_0926.pth")
    # model_path_dp = Path("/home/weihanx/videogpt/workspace/transformer_prior/model_True_prior_0926.pth")
    # model_config(model_path_dp)
    # decode_method = "beam_search_window"
    # sentence = True
    # diffusion_prior = False
    # # test_phase_zeroshot(decode_method, sentence, diffusion_prior, device = "cuda:0")
    # checkpoint_path = '/home/weihanx/videogpt/workspace/transformer_prior/model_False_prior_0926.pth'
    # checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # # Assuming the checkpoint contains the state_dict of the model
    # state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint

    # # Calculate the total number of parameters
    # total_params = sum(p.numel() for p in state_dict.values())

    # print(total_params)