import contextlib
from functools import lru_cache
from typing import Optional, TypedDict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pyannote.core.utils.generators import pairwise
from pyannote.audio.core.model import Model
from pyannote.audio.core.task import Task
from pyannote.audio.utils.params import merge_dict
from pyannote.audio.utils.receptive_field import (
    conv1d_num_frames,
    conv1d_receptive_field_center,
    conv1d_receptive_field_size,
)
from pyannote.core import SlidingWindow
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from modules.conbimamba_3conv import ConBiMambaEncoder, Linear

class ConBiMambaParams(TypedDict):
    input_dim: int
    encoder_dim: int
    num_layers: int
    num_attention_heads: int
    feed_forward_expansion_factor: int
    conv_expansion_factor: int
    input_dropout_p: float
    feed_forward_dropout_p: float
    attention_dropout_p: float
    conv_dropout_p: float
    conv_kernel_sizes: list 
    half_step_residual: bool
    merge_mode: str

def build_blocks_from_cfg(params: ConBiMambaParams, in_features: int) -> tuple[nn.Module, int]:
    conbimamba = ConBiMambaEncoder(
        input_dim=in_features,
        encoder_dim=params["encoder_dim"],
        num_layers=params["num_layers"],
        num_attention_heads=params["num_attention_heads"],
        feed_forward_expansion_factor=params["feed_forward_expansion_factor"],
        conv_expansion_factor=params["conv_expansion_factor"],
        input_dropout_p=params["input_dropout_p"],
        feed_forward_dropout_p=params["feed_forward_dropout_p"],
        attention_dropout_p=params["attention_dropout_p"],
        conv_dropout_p=params["conv_dropout_p"],
        conv_kernel_sizes=params["conv_kernel_sizes"],
        half_step_residual=params["half_step_residual"],
        merge_mode=params["merge_mode"]
    )
    return conbimamba, params["encoder_dim"]

class LWFA5_conbimamba(Model):

    WAV2VEC_DEFAULTS = "WAVLM_BASE"
    LINEAR_DEFAULTS = {"hidden_size": 128, "num_layers": 2}
    CONBIMAMBA_DEFAULTS: ConBiMambaParams = {
        "input_dim": 768,
        "encoder_dim": 256,
        "num_layers": 6,
        "num_attention_heads": 8,
        "feed_forward_expansion_factor": 2,
        "conv_expansion_factor": 2,
        "input_dropout_p": 0.1,
        "feed_forward_dropout_p": 0.1,
        "attention_dropout_p": 0.1,
        "conv_dropout_p": 0.1,
        "conv_kernel_sizes": [15, 31, 63],
        "half_step_residual": True,
        "merge_mode": "avg"
    }

    def __init__(
        self,
        wav2vec: Optional[Union[dict, str]] = None,
        wav2vec_layer: int = -1,
        conbimamba: Optional[ConBiMambaParams] = None,
        linear: Optional[dict] = None,
        sample_rate: int = 16000,
        num_channels: int = 1,
        task: Optional[Task] = None,
    ):
        super().__init__(sample_rate=sample_rate, num_channels=num_channels, task=task)

        if wav2vec is None:
            wav2vec = self.WAV2VEC_DEFAULTS

        if isinstance(wav2vec, str):
            if hasattr(torchaudio.pipelines, wav2vec):
                bundle = getattr(torchaudio.pipelines, wav2vec)
                if sample_rate != bundle._sample_rate:
                    raise ValueError(f"Expected {bundle._sample_rate}Hz, found {sample_rate}Hz.")
                wav2vec_dim: int = bundle._params["encoder_embed_dim"]
                wav2vec_num_layers = bundle._params["encoder_num_layers"]
                self.wav2vec = bundle.get_model()
                if hasattr(self.wav2vec, "model"):
                    self.wav2vec = self.wav2vec.model
                print("Wav2Vec type:", type(self.wav2vec))
                print("Wav2Vec attributes:", dir(self.wav2vec))
            else:
                _checkpoint = torch.load(wav2vec)
                wav2vec_config = _checkpoint.pop("config")
                self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec_config)
                state_dict = _checkpoint.pop("state_dict")
                self.wav2vec.load_state_dict(state_dict)
                wav2vec_dim: int = wav2vec_config["encoder_embed_dim"]
                wav2vec_num_layers = wav2vec_config["encoder_num_layers"]
        elif isinstance(wav2vec, dict):
            self.wav2vec = torchaudio.models.wav2vec2_model(**wav2vec)
            wav2vec_dim: int = wav2vec["encoder_embed_dim"]
            wav2vec_num_layers = wav2vec["encoder_num_layers"]
        else:
            raise ValueError("Expected `wav2vec` to be a string, a dictionary, or a path to a checkpoint.")

        if not hasattr(self.wav2vec, "feature_extractor"):
            raise AttributeError(
                f"Wav2Vec model ({type(self.wav2vec)}) does not have 'feature_extractor' attribute. "
                "Check torchaudio version or model configuration."
            )

        receptive_field_size = self.receptive_field_size(num_frames=1)
        receptive_field_step = (
            self.receptive_field_size(num_frames=2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center(frame=0) - (receptive_field_size - 1) / 2
        )
        self.receptive_field = SlidingWindow(
            start=receptive_field_start / self.hparams.sample_rate,
            duration=receptive_field_size / self.hparams.sample_rate,
            step=receptive_field_step / self.hparams.sample_rate,
        )

        if wav2vec_layer < 0:
            self.wav2vec_weights = nn.Parameter(data=torch.ones(wav2vec_num_layers), requires_grad=True)

        conbimamba: ConBiMambaParams = merge_dict(self.CONBIMAMBA_DEFAULTS, conbimamba)
        linear = merge_dict(self.LINEAR_DEFAULTS, linear)

        self.save_hyperparameters("wav2vec", "wav2vec_layer", "conbimamba", "linear")

        self.conbimamba, conbimamba_outdim = build_blocks_from_cfg(conbimamba, wav2vec_dim)

        self.conbimamba_layer_weights = nn.Parameter(torch.ones(conbimamba["num_layers"]), requires_grad=True)
        self.fusion_norm = nn.LayerNorm(conbimamba["encoder_dim"])
        self.fusion_dropout = nn.Dropout(p=0.1)

        self.change_point_head = nn.Sequential(
            Linear(128, 64),
            nn.ReLU(),
            Linear(64, 1)
        )

        if linear["num_layers"] < 1:
            self.linear = None
        else:
            self.linear = nn.ModuleList(
                [
                    nn.Linear(in_features, out_features)
                    for in_features, out_features in pairwise(
                        [conbimamba_outdim] + [linear["hidden_size"]] * linear["num_layers"]
                    )
                ]
            )

        self.freeze_wavlm = True

    @property
    def dimension(self) -> int:
        if isinstance(self.specifications, tuple):
            raise ValueError("does not support multi-tasking.")
        if self.specifications.powerset:
            return self.specifications.num_powerset_classes
        else:
            return len(self.specifications.classes)

    def build(self):
        if self.hparams.linear["num_layers"] > 0:
            in_features = self.hparams.linear["hidden_size"]
        else:
            in_features = self.hparams.conbimamba["encoder_dim"]

        self.classifier = Linear(in_features, self.dimension, bias=False)
        self.activation = self.default_activation()

    @lru_cache
    def num_frames(self, num_samples: int) -> int:
        num_frames = num_samples
        for conv_layer in self.wav2vec.feature_extractor.conv_layers:
            num_frames = conv1d_num_frames(
                num_frames,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return num_frames

    def receptive_field_size(self, num_frames: int = 1) -> int:
        receptive_field_size = num_frames
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_size = conv1d_receptive_field_size(
                num_frames=receptive_field_size,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_size

    def receptive_field_center(self, frame: int = 0) -> int:
        receptive_field_center = frame
        for conv_layer in reversed(self.wav2vec.feature_extractor.conv_layers):
            receptive_field_center = conv1d_receptive_field_center(
                receptive_field_center,
                kernel_size=conv_layer.kernel_size,
                stride=conv_layer.stride,
                padding=conv_layer.conv.padding[0],
                dilation=conv_layer.conv.dilation[0],
            )
        return receptive_field_center

    def forward(self, waveforms: torch.Tensor, input_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_layers = None if self.hparams.wav2vec_layer < 0 else self.hparams.wav2vec_layer

        with torch.no_grad() if self.freeze_wavlm else contextlib.nullcontext():
            outputs, _ = self.wav2vec.extract_features(waveforms.squeeze(1), num_layers=num_layers)

        if num_layers is None:
            outputs = torch.stack(outputs, dim=-1) @ F.softmax(self.wav2vec_weights, dim=0)
        else:
            outputs = outputs[-1]

        if input_lengths is not None:
            output_lengths = input_lengths >> 2
            output_lengths = output_lengths - 1
        else:
            output_lengths = torch.tensor([outputs.size(1)] * outputs.size(0), device=outputs.device)

        layer_outputs, _, _ = self.conbimamba(outputs, output_lengths)
        stacked = torch.stack(layer_outputs, dim=-1)
        device1 = self.conbimamba_layer_weights.device
        mask = torch.tensor([0, 0, 1, 1, 1, 1, 1], dtype=torch.float32, device=device1)
        masked_weights = self.conbimamba_layer_weights.clone()
        masked_weights = masked_weights.masked_fill(mask == 0, float('-1e9'))
        weights = F.softmax(masked_weights, dim=0)
        outputs = torch.einsum("btdl,l->btd", stacked, weights)

        outputs = self.fusion_norm(outputs)
        outputs = self.fusion_dropout(outputs)

        if self.linear is not None:
            for linear in self.linear:
                outputs = F.leaky_relu(linear(outputs))

        main_output = self.activation(self.classifier(outputs))  # [batch, time, classes]

        change_point_output = torch.sigmoid(self.change_point_head(outputs))  # [batch, time, 1]

        return main_output, change_point_output