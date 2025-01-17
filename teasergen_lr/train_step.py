import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel, AutoTokenizer
from torchvision import models
from dataset import SentenceImageDataset
import pathlib 
import tqdm
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
import wandb
# from SSIM_Loss import ssim_loss
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
    sample_wise_sim_mean = []
    for i in range(bsz):
        sample_wise_sim = []

        for j in range(seq_len):
            # shape matrxi1[i][j] = 768
            l2_sim = F.pairwise_distance(matrix1[i][j].unsqueeze(0), matrix2[i][j].unsqueeze(0), p=2)
            sample_wise_sim.append(l2_sim.squeeze().item())
        # print(f"elemnt = {i} in current batch, element-wise difference = {sample_wise_sim}")
        sample_wise_sim_mean.append(sum(sample_wise_sim)/len(sample_wise_sim))
    return sum(sample_wise_sim_mean)/len(sample_wise_sim_mean) # this batch similarity
        
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
    combined_dict_main = load_json(paths['train_annotate_main']) # smaller set, common video name not text name
    train_eval_list_main = list(combined_dict_main.keys()) 
    combined_dict_intro = load_json(paths['train_annotate_intro']) # smaller set
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

def setup_dataloader(datasets, batch_size, jobs):
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=jobs, collate_fn=SentenceImageDataset.collate_fn),
        'valid': DataLoader(datasets['eval'], batch_size=batch_size, shuffle=True, num_workers=jobs, collate_fn=SentenceImageDataset.collate_fn)
    }

def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.

    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.

    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position



def train(apply_diffusion_prior, type_loss, device, narrator_only, out_dir):
    """Main function."""
    if type_loss == "l2":
        criterion = nn.MSELoss()
    elif type_loss == "vgg_perceptual":
        criterion = vgg_loss.WeightedLoss([vgg_loss.VGGLoss(shift=2),
                                    nn.MSELoss(),
                                    vgg_loss.TVLoss(p=1)],
                                    [1, 40, 10]).to(device)

    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    # # Get the specified device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = setup_paths(narrator_only)
    wandb.init(project="transformer_prior", name=f"diffusion_prior_{apply_diffusion_prior}_narr_only_{narrator_only}_l2_total_steps")
    train_list, eval_list = data_split(paths)
    lr_warmup_steps = 5000
    lr_decay_steps = 10000
    lr_decay_multiplier = 0.1
    jobs = 4
    grad_norm_clip = 1
    grad_acc_steps = 4
    batch_size = 4 
    target_steps = 50000
    learning_rate = 1e-4
    weight_decay = 1e-2 
    valid_steps = 500
    wandb.config = {
    "apply_transformer_prior": apply_diffusion_prior,
    "steps": target_steps,
    "actual_batch_size": batch_size * grad_acc_steps,
    "narrator_only": narrator_only,
    "lr_warmup_steps": lr_warmup_steps, 
    "lr_decay_steps": lr_decay_steps,
    "lr_decay_multiplier": lr_decay_multiplier,
    "grad_norm_clip":grad_norm_clip,
    "num_jobs": jobs,
    "learning_rate":learning_rate,
    "weight_decay":weight_decay
    }
    datasets = create_datasets(paths, train_list, eval_list)
    print(f"Creating the data loader...")
    train_loader = setup_dataloader(datasets, batch_size, jobs)['train']
    valid_loader = setup_dataloader(datasets, batch_size, jobs)['valid']

    print(f"Creating model...")
    model = model.to(device)

    # Create the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )


    # Initialize variables
    step = 0

    train_iterator = iter(train_loader)
    # Wrap with a try-except block to handle keyboard interrupt
    try:
        while step < target_steps:
            model.train()
            recent_losses = []

            # Clear gradients
            optimizer.zero_grad()


            # Training loop
            pbar = tqdm.tqdm(total=valid_steps, ncols=120)
            for local_step in range(valid_steps * grad_acc_steps):
                # Get next batch
                try:
                    batch = next(train_iterator)
                except StopIteration:
                    # Reinitialize dataset iterator
                    train_iterator = iter(train_loader)
                    batch = next(train_iterator)

                sentences, images, mask, docu_name, sentence_text = batch

                # print(f"sentences shape: {sentences.shape}, images shape: {images.shape}, mask shape{mask.shape}")
                sentences = sentences.to(device)
                images = images.to(device)
                mask = mask.to(device)

                transformer_output = model(sentences, mask, sentence_text) # batch size, seq_len, 768
                # need to flip the mask
                flipped_mask = ~mask.bool()
                mask_expanded = flipped_mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
                
                masked_transformer_output = transformer_output * mask_expanded
                # positive_pairs, negative_pairs = construct_pairs_optimized(masked_transformer_output, scene_batch)
                # contrastive_loss = custom_contrastive_loss(positive_pairs, negative_pairs, margin=1.0)
                transformer_output_sum = masked_transformer_output.sum(dim=1)
                non_padded_count = flipped_mask.sum(dim=1, keepdim=True).clamp(min=1)  # those 1 should be valid count
                transformer_output_mean = transformer_output_sum / non_padded_count

                masked_images = images * mask_expanded  

                image_embeddings_sum = masked_images.sum(dim=1)
                image_non_padded_count = flipped_mask.sum(dim=1, keepdim=True).clamp(min=1)
                image_embeddings_mean = image_embeddings_sum / image_non_padded_count
                # average mean: batch size, dimension
                # print(f"transformer_output_mean shape = {transformer_output_mean.shape}, image_embeddings_mean shape = {image_embeddings_mean.shape}")
                loss = criterion(transformer_output_mean, image_embeddings_mean)
                
                loss = loss / grad_acc_steps # divide by accumaulation steps
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_norm_clip
                )
                if (local_step + 1) % grad_acc_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    step += 1
                    pbar.update(1)

                # Compute the moving average of the loss
                recent_losses.append(float(loss) * grad_acc_steps) # total loss for that batch
                if len(recent_losses) > 10:
                    del recent_losses[0]
                train_loss = np.mean(recent_losses)
                wandb.log({"moving average of train_loss": train_loss})
                pbar.set_postfix(loss=f"{train_loss:8.4f}")
            pbar.close()

            # Release GPU memory right away
            del seq, mask

            # Save the model
            checkpoint_filename = checkpoint_dir / f"model_{step}.pt"
            torch.save(model.state_dict(), checkpoint_filename)

            model.eval() # stop training
            with torch.no_grad():
                for batch in valid_loader:
                    sentences, images, mask, docu_name, sentence_text = batch

                    sentences = sentences.to(device)
                    images = images.to(device)
                    mask = mask.to(device)

                    transformer_output = model(sentences, mask, sentence_text)
                    # flip the mask
                    flipped_mask = ~mask.bool()
                    mask_expanded = flipped_mask.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]
                    masked_transformer_output = transformer_output * mask_expanded

                    transformer_output_sum = masked_transformer_output.sum(dim=1)
                    non_padded_count = flipped_mask.sum(dim=1, keepdim=True).clamp(min=1)  # Avoid division by zero
                    transformer_output_mean = transformer_output_sum / non_padded_count

                    masked_images = images * mask_expanded  

                    image_embeddings_sum = masked_images.sum(dim=1)
                    image_non_padded_count = flipped_mask.sum(dim=1, keepdim=True).clamp(min=1)
                    image_embeddings_mean = image_embeddings_sum / image_non_padded_count

                    loss = criterion(transformer_output_mean, image_embeddings_mean)
                    wandb.log({"batch valid_loss": loss})

            # Release GPU memory right away
            del seq, mask

    except KeyboardInterrupt:
        print("Detected KeyboardInterrupt and stopped the training!")
