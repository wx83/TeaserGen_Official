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

def setup_dataloader(datasets, batch_size):
    return {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=SentenceImageDataset.collate_fn),
        'valid': DataLoader(datasets['eval'], batch_size=batch_size, shuffle=True, collate_fn=SentenceImageDataset.collate_fn)
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


def train(apply_diffusion_prior, type_loss, device, narrator_only, output_dir): 
    paths = setup_paths(narrator_only)
    wandb.init(project="transformer_prior", name=f"diffusion_prior_{apply_diffusion_prior}_narr_only_{narrator_only}_l2")
    train_list, eval_list = data_split(paths)
    lr_warmup_steps = 5000
    lr_decay_steps = 10000
    lr_decay_multiplier = 0.1
    grad_norm_clip = 1
    wandb.config = {
    "apply_transformer_prior": apply_diffusion_prior,
    "epochs": 15,
    "batch_size": 4,
    "narrator_only": narrator_only,
    "lr_warmup_steps": 5000, 
    "lr_decay_steps": 10000,
    "lr_decay_multiplier": 0.1,
    "grad_norm_clip":1,
    }
    datasets = create_datasets(paths, train_list, eval_list)
    train_dataloader = setup_dataloader(datasets, 4)['train']
    valid_dataloader = setup_dataloader(datasets, 4)['valid']

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
    num_epochs = 15
    batch_wise_training_loss = []
    batch_wise_val_loss = []
    
    # # Training loop
    # from time import time
    # set gradient accumulate
    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Zero gradients at the start of the epoch
        epoch_loss = 0.0
        batch_similarity_1_list = []
        batch_similarity_2_list = []
        batch_similarity_3_list = []
        for i, batch in enumerate(train_dataloader):
            # print(f"Processing batch")
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
            optimizer.zero_grad()   # clear last step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), grad_norm_clip
            )
            optimizer.step()  # Perform a single optimization step

            # Accumulate the loss for this epoch
            epoch_loss += loss.item()
        # Print loss for the epoch
            masked_input = sentences * mask_expanded

            batch_similarity_1 = check_similarity(masked_transformer_output, masked_images)
            batch_similarity_2 = check_similarity(masked_input, masked_images) # how far is 
            batch_similarity_3 = check_similarity(masked_transformer_output, masked_input)
            batch_similarity_1_list.append(batch_similarity_1)
            batch_similarity_2_list.append(batch_similarity_2)
            batch_similarity_3_list.append(batch_similarity_3)
            batch_wise_training_loss.append(loss)
            wandb.log({"batch_similarity_1": batch_similarity_1, "batch_similarity_2": batch_similarity_2, "batch_similarity_3": batch_similarity_3, "batch_wise_training_loss": loss})
        model.eval()

        val_loss = 0
        with torch.no_grad():  # Essential for efficiency
            val_loss = 0
            for batch in valid_dataloader:
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
                val_loss += loss.item()
                batch_wise_training_loss.append(loss)
                wandb.log({"batch_wise_val_loss": loss})
        if epoch % 2 == 0:
            output_path = output_dir / f"checkpoint_epoch_{epoch}.pth"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_path)
    # save the batch loss as npy
    batch_wise_training_loss = np.array(batch_wise_training_loss)
    batch_wise_val_loss = np.array(batch_wise_val_loss)
    np.save(output_dir / "batch_wise_training_loss.npy", batch_wise_training_loss)
    np.save(output_dir / "batch_wise_val_loss.npy", batch_wise_val_loss)
    end_time = time.time()
    print(f"training one epoch need = {end_time - start_time}")
    print("Training complete.")
    wandb.finish()



if __name__ == "__main__":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loss  "ssim":
    # train(apply_diffusion_prior=False, type_loss="vgg_perceptual")
    # /home/weihanx/videogpt/workspace/transformer_prior/model_False_1_l2_prior_0103.pth
    # body_contents_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/test_embedding/clip_frame_768_emb_main")
    apply_diffusion_prior = False
    narrator_only = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(f"/home/weihanx/videogpt/workspace/transformer_prior/model/diffusion_prior_{apply_diffusion_prior}/narr_only_{narrator_only}/{time.strftime('%m%d')}")
    output_dir.mkdir(parents=True, exist_ok=True)
    train(apply_diffusion_prior=apply_diffusion_prior, type_loss="l2", device = device, narrator_only = narrator_only, output_dir=output_dir) # retrain by fix bug
   
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