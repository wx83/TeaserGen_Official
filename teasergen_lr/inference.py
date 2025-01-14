from decoding import *

from transformer import CustomModel
from dataset import SentenceImageDataset
from torch.utils.data import DataLoader
from pathlib import Path
import time
import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader

def load_model(apply_diffusion_prior, model_dir_path, device):
    model_dir = Path(model_dir_path)
    model = CustomModel(apply_diffusion_prior, device=device).to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    return model

def prepare_data(test_annotate_dir, sentence_emb_dir, sentence_json_intro_dir):
    test_annotate = Path(test_annotate_dir)
    combined_dict_test = load_json(test_annotate)
    test_list = list(combined_dict_test.keys())

    sentence_json_intro = Path(sentence_json_intro_dir)
    sentence_emb_dir = Path(sentence_emb_dir)
    image_emd_dir = None  # Assuming no image embedding directory is needed
    
    test_dataset = SentenceImageDataset(
        test_list, test_annotate, image_emd_dir, sentence_emb_dir, sentence_json_intro
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=True, collate_fn=SentenceImageDataset.collate_fn
    )
    return test_dataloader

def run_test(apply_diffusion_prior, model_dir_path, device, narration_type, decode_method, smooth, body_contents_dir, output_dir):
    test_dataloader = prepare_data(
        "/home/weihanx/videogpt/workspace/transformer_prior/test_gpt.json",
        "/home/weihanx/videogpt/deepx_data6/clip_sentence_emb_768_average",
        "/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json"
    )

    model = load_model(apply_diffusion_prior, model_dir_path, device)

    start = time.time()
    for batch in test_dataloader:
        sentences, images, mask, docu_name_list, sentence_text = batch

        sentences = sentences.to(device)
        images = images.to(device)
        mask = mask.to(device)

        transformer_output = model(sentences, mask, sentence_text)
        output_embedding = transformer_output  # Apply mask

        smooth_dir = "smooth" if smooth else "no_smooth"
        method_directory = 'greedy' if decode_method == 'greedy' else 'beam_search_window'
        output_path = Path(output_dir) / narration_type / method_directory / smooth_dir
        output_path.mkdir(parents=True, exist_ok=True)
        sentence_list_json = Path("/home/weihanx/videogpt/workspace/transformer_prior/test_gpt_sentence_list.json")
        if decode_method == "greedy":
            decode_greedy_sentence(1, output_embedding, mask, docu_name_list, Path(body_contents_dir), output_path, 1)
        elif decode_method == "beam_search_window":
            decode_beam_search_history(1, output_embedding, mask, docu_name_list, Path(body_contents_dir), 5, 10, output_path, 1, sentence_list_json, 1)

    end = time.time()
    print(f"Time: {end - start}")
    

def model_config(model_path):
    model_state_dict = torch.load(model_path)

    # Count the number of parameters in the model
    total_params = sum(p.numel() for p in model_state_dict.values())
    print(f"total params = {total_params}")

if __name__ == "__main__":
    pass
