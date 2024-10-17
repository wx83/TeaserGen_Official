import numpy as np
import torch
import torch.nn.functional as F
from importlib import import_module
from identity_separation_model import IdentitySeparationModel
import os
import yaml
from aimless.utils import MWF
from torchaudio.transforms import Spectrogram, InverseSpectrogram


def load_checkpoint(ckpt_path, ckpt_name):
    with open(os.path.join(ckpt_path, "config.yaml")) as f:
        config = yaml.safe_load(f)

    ckpt_path = os.path.join(ckpt_path, "checkpoints", ckpt_name)

    core_model_configs = config["model"]["init_args"]["model"]
    module_path, class_name = core_model_configs["class_path"].rsplit(".", 1)
    module = import_module(module_path)
    model = getattr(module, class_name)(**core_model_configs["init_args"])

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = {}
    for k, v in checkpoint["state_dict"].items():
        if k.startswith("model."):
            state_dict[k.replace("model.", "")] = v
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


class EnsembleNet(IdentitySeparationModel):
    """
    Doesn't do any separation just passes the input back as output
    """

    hdemucs_path = "./lightning_logs/hdemucs-64-sdr/"
    music_bandsplitRNN_path = "./lightning_logs/bandsplitRNN-music/"
    speech_bandsplitRNN_path = "./lightning_logs/bandsplitRNN-speech/"
    hdemucs_ckpt_name = "last.ckpt"
    bandsplitRNN_ckpt_name = "last.ckpt"
    
    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Ensemble.py: running on device = {device}")
    instruments_idx = {
        "dialog": 2,
        "effect": 1,
        "music": 0,
    }

    def __init__(self):
        super().__init__()
        model, config = load_checkpoint(self.hdemucs_path, self.hdemucs_ckpt_name)
        self.hdemucs = model.to(self.device)

        model, config = load_checkpoint(
            self.music_bandsplitRNN_path, self.bandsplitRNN_ckpt_name
        )
        self.rnn_music = model.to(self.device)

        # model, config = load_checkpoint(
        #     self.speech_bandsplitRNN_path, self.bandsplitRNN_ckpt_name
        # )
        # self.rnn_speech = model.to(self.device)

        n_fft = config["model"]["init_args"]["n_fft"]
        hop_length = config["model"]["init_args"]["hop_length"]

        self.spec = Spectrogram(n_fft=n_fft, hop_length=hop_length, power=None).to(
            self.device
        )
        self.inv_spec = InverseSpectrogram(n_fft=n_fft, hop_length=hop_length).to(
            self.device
        )
        self.mwf = MWF(residual_model=True, softmask=False, n_iter=1).to(self.device)

    @torch.no_grad()
    def separate_music_file(self, mixed_sound_array, sample_rate):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """
        mixed_sound_array = (
            torch.from_numpy(mixed_sound_array.T).float().to(self.device).unsqueeze(1)
        )
        mixed_mean = mixed_sound_array.mean(dim=0)
        mixed_diff = 0.5 * (mixed_sound_array[0] - mixed_sound_array[1])
        mixed_sound_array = torch.cat(
            [
                mixed_sound_array,
                mixed_diff.unsqueeze(0),
                mixed_mean.unsqueeze(0),
            ],
            dim=0,
        )

        separated_music_arrays = {}
        output_sample_rates = {}

        sep_l, sep_r, sep_diff, sep_mean = self.hdemucs(mixed_sound_array).squeeze()

        dialog_idx = self.instruments_idx["dialog"]
        dialog = (sep_l[dialog_idx] + sep_r[dialog_idx] + sep_mean[dialog_idx]) / 3
        separated_music_arrays["dialog"] = (
            dialog.unsqueeze(1).repeat(1, 2).cpu().numpy()
        )
        output_sample_rates["dialog"] = sample_rate

        mixed_mean[0] = mixed_mean[0] - dialog

        effect_idx = self.instruments_idx["effect"]
        music_idx = self.instruments_idx["music"]
        effect_diff = mixed_diff[0] - sep_diff[music_idx]
        effect_l = effect_diff + sep_mean[effect_idx]
        effect_r = sep_mean[effect_idx] - effect_diff
        effect_l = 0.5 * (effect_l + sep_l[effect_idx])
        effect_r = 0.5 * (effect_r + sep_r[effect_idx])

        separated_music_arrays["effect"] = (
            torch.stack([effect_l, effect_r], 1).cpu().numpy()
        )
        output_sample_rates["effect"] = sample_rate

        music_diff = mixed_diff[0] - sep_diff[effect_idx]
        music_l = music_diff + sep_mean[music_idx]
        music_r = sep_mean[music_idx] - music_diff
        music_l = 0.5 * (music_l + sep_l[music_idx])
        music_r = 0.5 * (music_r + sep_r[music_idx])
        separated_music_arrays["music"] = (
            torch.stack([music_l, music_r], 1).cpu().numpy()
        )
        output_sample_rates["music"] = sample_rate

        # bandsplitRNN
        X = self.spec(mixed_sound_array[:2].transpose(0, 1))
        music_mask = self.rnn_music(X.abs())
        # speech_mask = self.rnn_speech(X.abs())
        # mask = torch.stack([music_mask, speech_mask], dim=1)
        mask = music_mask.unsqueeze(1)
        # speech_spec = X * mask
        speech_spec = self.mwf(mask, X)
        separated = self.inv_spec(speech_spec).squeeze().cpu()

        if separated.shape[-1] < mixed_sound_array.shape[-1]:
            separated = F.pad(
                separated, (0, mixed_sound_array.shape[-1] - separated.shape[-1])
            )

        separated_music, separated_fx = separated.permute(0, 2, 1).numpy()

        # input_length = len(left_mixed_arr)
        separated_music_arrays["music"] += separated_music
        separated_music_arrays["music"] /= 2

        # separated_music_arrays["effect"] += separated_fx
        # separated_music_arrays["effect"] /= 2

        # separated_music_arrays["dialog"] += separated_speech.mean(axis=1, keepdims=True)
        # separated_music_arrays["dialog"] /= 2

        return separated_music_arrays, output_sample_rates


class HDemucsMWF(EnsembleNet):
    n_fft = 4096
    hop_length = 1024

    def __init__(self):
        super().__init__()
        self.spec = Spectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length, power=None
        ).to(self.device)
        self.inv_spec = InverseSpectrogram(
            n_fft=self.n_fft, hop_length=self.hop_length
        ).to(self.device)
        self.mwf = MWF(softmask=True).to(self.device)

    @torch.no_grad()
    def separate_music_file(self, mixed_sound_array, sample_rate):
        mixed_sound_array = (
            torch.from_numpy(mixed_sound_array.T).float().to(self.device).unsqueeze(1)
        )
        # B, C, S, T
        seperated = self.hdemucs(mixed_sound_array).transpose(0, 2)
        # B, S, C, T
        mask_hat = self.spec(seperated).abs()
        # B, S, C, F, T
        mix_spec = self.spec(mixed_sound_array.transpose(0, 1))
        # B, C, F, T

        seperated = self.inv_spec(self.mwf(mask_hat, mix_spec)).squeeze().cpu()
        if seperated.shape[-1] < mixed_sound_array.shape[-1]:
            seperated = F.pad(
                seperated, (0, mixed_sound_array.shape[-1] - seperated.shape[-1])
            )

        seperated = seperated.transpose(1, 2).numpy()

        # input_length = len(left_mixed_arr)
        separated_music_arrays = {}
        output_sample_rates = {}

        for instrument in self.instruments:
            separated_music_arrays[instrument] = seperated[
                self.instruments_idx[instrument]
            ]
            output_sample_rates[instrument] = sample_rate

        return separated_music_arrays, output_sample_rates


