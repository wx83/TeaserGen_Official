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
from inference import *
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

# SSIM Loss

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Ensure x is a tensor of shape (batch_size, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x



class CustomModel(nn.Module):
    def __init__(self, apply_diffusion_prior, hidden_dim=768, nhead=12, num_layers=6, device="cuda:0"):
        super(CustomModel, self).__init__()
        
        # Load frozen sentence transformer model
        # should load the 
        # self.sentence_transformer = AutoModel.from_pretrained(sentence_model_name)
        # for param in self.sentence_transformer.parameters():
        #     param.requires_grad = False
        # already freeze to get embeddings
        # Position encoding for the transformer encoder
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
        # print(f"unique_indceis = {map_mean_list}")
        # List to store the averaged embeddings
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
    
    def get_dp(self, sentences, device="cuda:0"):
        map_mean_list = []
        all_sentences = []

        print(f"num of sentences in this sample = {len(sentences)}")
        for idx, sub_sent in enumerate(sentences):
            try:
                tokenized_texts = clip.tokenize(sub_sent, context_length=77).to(device)
                map_mean_list.append(idx)
                all_sentences.append(sub_sent)
            except Exception as e:
                print(f"not able to process long sentence")
                sub_sentences = re.split(r'[，。！？,.;:]', sub_sent)
                sub_sentences = [s.strip() for s in sub_sentences if s.strip()]
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
        # Step 1: Pass sentences through the frozen sentence transformer
        # sentence_embeddings = self.sentence_transformer(**sentences).last_hidden_state
        # load sentence embedding
        # sentence_emb: # sentence * embeding, 768
        # image_emb: # frame * embding, 1024
        # diffusion prior
        if self.apply_diffusion_prior == True:
            for b in range(len(sentence_text)):
                # Batch process sentences in sentence_text[b]
                predicted_embedding_list = self.get_dp(sentence_text[b], self.device)  # Now handles batch input
                # No need to use a loop; predicted_embedding_list is already in the form (num_sentences, 768)
                total_len_dp = predicted_embedding_list.shape[0]
                # print(f"predicted shpae = {predicted_embedding_list.shape}")
                # Add predicted embeddings to sentence_embeddings
                sentence_embeddings[b][:total_len_dp] += predicted_embedding_list
        # print(f"sentence embedidng into transfomre = {sentence_embeddings.shape}")
        sentence_embeddings = self.position_encoding(sentence_embeddings)
        # print(f"sentence_embeddings shape: {sentence_embeddings.shape}") 4,45,768
        # Step 2: Pass sentence embeddings through the transformer encoder
        # this is output embedding
        transformer_output = self.transformer_encoder(sentence_embeddings, src_key_padding_mask=mask)
        # add trainable group embedding
        # output_embedding = transformer_output + self.scene_embedding # add a group embeeding
        
        return transformer_output

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

    # Loss for negative pairs: we want to minimize the mean_neg_sim (i.e., maximize (mean_neg_sim - margin))
    # Only apply the margin penalty if mean_neg_sim is less than margin
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
        
        # Compute all pairwise distances
        # Expand current sample to [seq_len, seq_len, dim] by broadcasting
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

def train(apply_diffusion_prior, type_loss, device): 

    # all saved at documentary/train_embedding
    image_emb_dir_intro = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_frame_768_emb_intro") # This is the embedding from teaser
    image_emb_dir_main = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_frame_768_emb_main") # This is the embedding from teaser
    sentence_emb_intro_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_text_768_intro")
    sentence_emb_main_dir = Path("/home/weihanx/videogpt/data_deepx/documentary/train_embedding/clip_text_768_main")
    train_annotate_intro = Path("/home/weihanx/videogpt/workspace/transformer_prior/intro/train_0103_intro_comb.json")
    train_annotate_main = Path("/home/weihanx/videogpt/workspace/transformer_prior/main/train_0103_main_comb.json")
    sentence_json_intro = Path("/home/weihanx/videogpt/workspace/transformer_prior/intro/intro_text_name.json") # why sentence dir
    sentence_json_main = Path("/home/weihanx/videogpt/workspace/transformer_prior/main/main_text_name.json")
    combined_dict_main = load_json(train_annotate_main) # smaller set
    train_eval_list_main = list(combined_dict_main.keys()) 
    combined_dict_intro = load_json(train_annotate_intro) # smaller set
    train_eval_list_intro = list(combined_dict_intro.keys())
    common_keys = list(set(train_eval_list_main).intersection(train_eval_list_intro))
    print(f"len of common keys = {len(common_keys)}")
    train_list = common_keys[:int(len(common_keys)*0.8)]
    eval_list = common_keys[int(len(common_keys)*0.8):]
    # can be combined or trained separately but # use the same set of filenames
    train_dataset_intro = SentenceImageDataset(train_list, train_annotate_intro, image_emb_dir_intro, sentence_emb_intro_dir, sentence_json_intro)
    train_dataset_main = SentenceImageDataset(train_list, train_annotate_main, image_emb_dir_main, sentence_emb_main_dir, sentence_json_main)
    # concatenate two dataset
    train_dataset = torch.utils.data.ConcatDataset([train_dataset_intro, train_dataset_main]) # should learn separately or together? They should not have temporal dependence
    bsz = 8
    val_dataset_intro = SentenceImageDataset(eval_list, train_annotate_intro, image_emb_dir_intro, sentence_emb_intro_dir, sentence_json_intro)
    val_dataset_main = SentenceImageDataset(eval_list, train_annotate_main, image_emb_dir_main, sentence_emb_main_dir, sentence_json_main)
    val_dataset = torch.utils.data.ConcatDataset([val_dataset_intro, val_dataset_main])
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=bsz,  # Set your desired batch size
        shuffle=True,  # Whether to shuffle the data at each epoch
        collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
    )
    print(f"len of train_loader = {len(train_dataloader)}")
    valid_dataloader = DataLoader(
        val_dataset,
        batch_size=bsz,  # Set your desired batch size
        shuffle=True,  # Whether to shuffle the data at each epoch
        collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
    )
    # Instantiate the model
    model = CustomModel(apply_diffusion_prior, device=device)
    model = model.to(device)

    # Define the loss function (e.g., Mean Squared Error for regression tasks)
    # criterion = nn.MSELoss()
    # vgg perceptial loss
    if type_loss == "l2":
        criterion = nn.MSELoss()
    elif type_loss == "vgg_perceptual":
        criterion = vgg_loss.WeightedLoss([vgg_loss.VGGLoss(shift=2),
                                    nn.MSELoss(),
                                    vgg_loss.TVLoss(p=1)],
                                    [1, 40, 10]).to(device)
    # elif type_loss == "ssim":
    #     criterion = SSIMLoss(alpha=0.5).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    # Number of epochs for training
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
            # print(f"documnet name: {docu_name}")
            # print(f"shape of scene batch = {scene_batch.shape}, should be batchszie, sentence shape[0]")
            # mask: batch size, seq len
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
    # save the model
        # get validation loss
        model.eval()
        val_loss = 0
        for batch in valid_dataloader:
            sentences, images, mask, docu_name, sentence_text = batch
            # print(f"documnet name: {docu_name}")
            # mask: batch size, seq len
            # print(f"sentences shape: {sentences.shape}, images shape: {images.shape}, mask shape{mask.shape}")
            sentences = sentences.to(device)
            images = images.to(device)
            mask = mask.to(device)
            # print(f"sentences shape: {sentences.shape}, images shape: {images.shape}")
            # Forward pass
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
            torch.save(model.state_dict(), f"/home/weihanx/videogpt/workspace/transformer_prior/model_{apply_diffusion_prior}_{epoch}_{type_loss}_prior_0103.pth")
        else:
            break
        # if epoch % 1 == 0:
        #     torch.save(model.state_dict(), f"/home/weihanx/videogpt/workspace/transformer_prior/model_{apply_diffusion_prior}_prior_0926.pth")
    end_time = time.time()
    print(f"training one epoch need = {end_time - start_time}")
    print("Training complete.")

def test_phase(narration_type, decode_method, sentence, apply_diffusion_prior, hidden_size, device = "cuda:0"):
    if hidden_size == 512:
        body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb")
    if hidden_size == 768:
        body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/eval_dataset/video_main_frame_emb_768")

    image_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_image_emb_full_768")
    if apply_diffusion_prior == True:
        # sentence_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_768_average") # should add during training
        model = CustomModel(apply_diffusion_prior,  device=device).to(device)
        model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model_True_prior_0916_2.pth")) # final result
        # model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model_True_prior_0916_2.pth")) # final result
        model.eval()

    if apply_diffusion_prior == False:
        # sentence_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_768_average")
        model = CustomModel(apply_diffusion_prior,  device=device).to(device)
        model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model_False_prior_0926.pth")) # final resutl: 0926
        # model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model.pth"))
        model.eval()

    if narration_type == "original":
        sentence_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_768_average")
        test_annotate = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_0915_comb_2.json")
        combined_dict_test = load_json(test_annotate)
        test_list = list(combined_dict_test.keys())
        sentence_dir = Path("/home/weihanx/videogpt/workspace/ori_narr_text")
        test_dataset =  SentenceImageDataset(test_list, test_annotate, sentence_emb_dir, image_emb_dir,sentence_dir)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,  # Set your desired batch size: if it is 1, it will not be padded
            shuffle=True,  # Whether to shuffle the data at each epoch
            collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
        )
    if narration_type == "gpt":
        sentence_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_768_average_gpt")
        test_annotate = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt.json")
        combined_dict_test = load_json(test_annotate)
        test_list = list(combined_dict_test.keys())
        sentence_dir = Path("/home/weihanx/videogpt/workspace/gpt_narr_text")
        test_dataset = SentenceImageDataset(test_list, test_annotate, sentence_emb_dir, image_emb_dir, sentence_dir)
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=1,  # Set your desired batch size: if it is 1, it will not be padded
            shuffle=True,  # Whether to shuffle the data at each epoch
            collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
        )

    start = time.time()
    for batch in test_dataloader:

        sentences, images, mask, docu_name_list, sentence_text = batch
    
        sentences = sentences.to(device)
        images = images.to(device)
        mask = mask.to(device)
        # sentence_dir = Path("/home/weihanx/videogpt/workspace/ori_narr_text")
        # print(f"sentence shape = {sentences.shape}")
        transformer_output = model(sentences, mask, sentence_text)
        output_embedding = transformer_output # [4,18,512], apply mask
        # print(f"output shape = {output_embedding.shape}")
        match_video_name = docu_name_list # len of 4x, my batch size is 1
        # if decode_method == "greedy" and narration_type == "original" and sentence == False and diffusion_prior == False:
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/ori_greedy_0926")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, 1)

        # if decode_method == "greedy" and narration_type == "original" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/ori_greedy_sentence_0926")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy_sentence(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, sentence_list_json, 1)
       
        if decode_method =="beam_search_window" and narration_type == "original" and sentence == True and diffusion_prior == False:
            print(f"processing... point3, sentence shape = {sentences.shape}, output shape = {transformer_output.shape}")
            sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_list.json")
            
            output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/ori_beamsearchwindow_0926_1")
            output_dir.mkdir(exist_ok=True, parents=True)
            decode_beam_search_history(1, output_embedding, mask, match_video_name, body_contents_dir, 5, 10, output_dir, 1 , sentence_list_json, 1)

        # if decode_method =="beam_search_window" and narration_type == "original" and sentence == True and diffusion_prior == True:
        #     print(f"processing... point3, sentence shape = {sentences.shape}, output shape = {transformer_output.shape}")
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/ori_beamsearchwindow_0918_1_dp")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_beam_search_history(1, output_embedding, mask, match_video_name, body_contents_dir, 5, 10, output_dir, 1, sentence_list_json, 1)
        

        # if decode_method == "greedy" and narration_type == "original" and sentence == False and diffusion_prior == True:
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/ori_greedy_0918_dp")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, 1)

        # if decode_method == "greedy" and narration_type == "original" and sentence == True and diffusion_prior == True:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_list_0905.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/ori_greedy_sentence_0926_dp")
        #     # time_duration_folder = Path("/home/weihanx/videogpt/deepx_data6/demo/demo_random")
        #     # demo_duration_output_dir = output_dir
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy_sentence(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, sentence_list_json, 1)
        

        # if decode_method =="beam_search_window" and narration_type == "gpt" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_beamsearchwindow_0917_1")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_beam_search_history(1, output_embedding, None, match_video_name, body_contents_dir, 5, 10, output_dir, 1, sentence_list_json, 1)


        # if decode_method =="beam_search_window" and narration_type == "gpt" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_beamsearchwindow_0917_point5")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_beam_search_history(1, output_embedding, None, match_video_name, body_contents_dir, 5, 10, output_dir, 0.5, sentence_list_json, 1)


        # if decode_method =="beam_search_window" and narration_type == "gpt" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_beamsearchwindow_0924_point5")
        #     output_dir.mkdir(exist_ok=True, parents=True)

        #     decode_beam_search_history(1, output_embedding, None, match_video_name, body_contents_dir, 5, 10, output_dir,0.5, sentence_list_json, 1)



        # if decode_method =="beam_search_window" and narration_type == "gpt" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_beamsearchwindow_0926_100_new")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_beam_search_history(1, output_embedding, None, match_video_name, body_contents_dir, 5, 10, output_dir, 100, sentence_list_json, 1)


        # if decode_method =="beam_search_window" and narration_type == "gpt" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_beamsearchwindow_0926_10_new")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_beam_search_history(1, output_embedding, None, match_video_name, body_contents_dir, 5, 10, output_dir, 10, sentence_list_json, 1)


        # if decode_method =="beam_search_window" and narration_type == "gpt" and sentence == True and diffusion_prior == True:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_beamsearchwindow_0926_1_new_2_dp")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_beam_search_history(1, output_embedding, None, match_video_name, body_contents_dir, 5, 10, output_dir, 1, sentence_list_json, 1)


        # if decode_method == "greedy" and narration_type == "gpt" and sentence == False and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_greedy_0926_new")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, 1)

        # if decode_method == "greedy" and narration_type == "gpt" and sentence == True and diffusion_prior == False:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_greedy_sentence_0926_new")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy_sentence(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, sentence_list_json, 1)

        # if decode_method == "greedy" and narration_type == "gpt" and sentence == False and diffusion_prior == True:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_greedy_0919_dp")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, 1)

        # if decode_method == "greedy" and narration_type == "gpt" and sentence == True and diffusion_prior == True:
        #     sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        #     output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/gpt_greedy_sentence_0926_dp")
        #     output_dir.mkdir(exist_ok=True, parents=True)
        #     decode_greedy_sentence(1, output_embedding, mask, match_video_name, body_contents_dir, output_dir, sentence_list_json, 1)

    end = time.time()
    print(f"Time: {end-start}")


def test_phase_zeroshot(decode_method, sentence, apply_diffusion_prior, device = "cuda:0"):
    body_contents_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshotvideo_clip_frame_768_emb")
    # image_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_image_emb_full_768") # This is useless for gpt inference
    image_emb_dir = None
    sentence_emb_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_clip_sentence_768_average")
    test_annotate = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_zeroshot.json")
    combined_dict_test = load_json(test_annotate)
    test_list = list(combined_dict_test.keys())
    sentence_dir = Path("/home/weihanx/videogpt/deepx_data6/zeroshot_narr_text") # clip_extractor_nodp
    sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_sentence_zeroshot.json")
    test_dataset = SentenceImageDataset(test_list, test_annotate, sentence_emb_dir, image_emb_dir, sentence_dir)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,  # Set your desired batch size: if it is 1, it will not be padded
        shuffle=True,  # Whether to shuffle the data at each epoch
        collate_fn=SentenceImageDataset.collate_fn  # Use the custom collate function
    )
    if apply_diffusion_prior == True:
        # sentence_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_768_average") # should add during training
        model = CustomModel(apply_diffusion_prior,  device=device).to(device)
        model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model_True_prior_0916_2.pth")) # final result
        # model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model_True_prior_0916_2.pth")) # final result
        model.eval()

    if apply_diffusion_prior == False:
        # sentence_emb_dir = Path("/home/weihanx/videogpt/workspace/clip_sentence_emb_768_average")
        model = CustomModel(apply_diffusion_prior,  device=device).to(device)
        model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model_False_prior_0926.pth")) # final resutl: 0926
        # model.load_state_dict(torch.load("/home/weihanx/videogpt/workspace/transformer_prior/model.pth"))
        model.eval()

    for batch in test_dataloader:

        sentences, images, mask, docu_name_list, sentence_text = batch
    
        sentences = sentences.to(device)
        images = images.to(device)
        mask = mask.to(device)
        # sentence_dir = Path("/home/weihanx/videogpt/workspace/ori_narr_text")
        print(f"sentence shape = {sentences.shape}")
        print(f"sentence_text = {len(sentence_text)}, len = {len(sentence_text[0])}")
        transformer_output = model(sentences, mask, sentence_text)
        output_embedding = transformer_output # [4,18,512], apply mask

        print(f"output shape = {output_embedding.shape}")
        match_video_name = docu_name_list # batch size is 1, no need for mask

        if decode_method =="beam_search_window" and sentence == True and diffusion_prior == False:
            print(f"processing... point3, sentence shape = {sentences.shape}, output shape = {transformer_output.shape}")
            output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/zeroshot_beamsearchwindow_1")
            output_dir.mkdir(exist_ok=True, parents=True)
            decode_beam_search_history(1, output_embedding, mask, match_video_name, body_contents_dir, 5, 10, output_dir, 1 , sentence_list_json, 1)

        if decode_method =="beam_search_window" and sentence == True and diffusion_prior == True:
            print(f"processing... point3, sentence shape = {sentences.shape}, output shape = {transformer_output.shape}")
            output_dir = Path("/home/weihanx/videogpt/workspace/transformer_prior/zeroshot_beamsearchwindow_1_dp")
            output_dir.mkdir(exist_ok=True, parents=True)
            decode_beam_search_history(1, output_embedding, mask, match_video_name, body_contents_dir, 5, 10, output_dir, 1 , sentence_list_json, 1)


def model_config(model_path):
    model_state_dict = torch.load(model_path)

    # Count the number of parameters in the model
    total_params = sum(p.numel() for p in model_state_dict.values())
    print(f"total params = {total_params}")
if __name__ == "__main__":
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # loss  "ssim":
    # train(apply_diffusion_prior=False, type_loss="vgg_perceptual")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train(apply_diffusion_prior=True, type_loss="l2", device = device) # retrain by fix bug
    # test_phase("gpt", "greedy", False, False, 768) # consider time interval
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