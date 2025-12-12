from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Make `div.*` importable without editing upstream import paths.
_ROOT = Path(__file__).resolve().parent
_ROOT_STR = str(_ROOT)
if _ROOT_STR not in sys.path:
    sys.path.insert(0, _ROOT_STR)

from div.backbones.bcd_nsf_bridge import NsfBcdBridge
from div.sdes import BridgeGAN


class BridgeVocInference(nn.Module):
    def __init__(
        self,
        hparams: Dict[str, Any],
        dnn_state_dict: Dict[str, torch.Tensor],
        infer_steps: Optional[int] = None,
    ):
        super().__init__()

        self.sampling_rate = int(hparams.get("sampling_rate", 44100))
        self.n_fft = int(hparams.get("n_fft", 2048))
        self.hop_size = int(hparams.get("hop_size", 512))
        self.win_size = int(hparams.get("win_size", 2048))
        self.fmin = float(hparams.get("fmin", 0.0))
        self.fmax = float(hparams.get("fmax", self.sampling_rate / 2))
        self.num_mels = int(hparams.get("num_mels", 128))

        self.spec_factor = float(hparams.get("spec_factor", 0.33))
        self.spec_abs_exponent = float(hparams.get("spec_abs_exponent", 0.5))
        self.transform_type = str(hparams.get("transform_type", "exponent"))
        self.drop_last_freq = bool(hparams.get("drop_last_freq", True))

        sde_kwargs = {k: hparams[k] for k in [
            "beta_min",
            "beta_max",
            "c",
            "k",
            "bridge_type",
            "N",
            "offset",
            "predictor",
            "sampling_type",
        ] if k in hparams}
        self.sde = BridgeGAN(**sde_kwargs)
        if infer_steps is not None:
            self.sde.N = int(infer_steps)

        dnn_kwargs = {k: hparams[k] for k in [
            "nblocks",
            "hidden_channel",
            "f_kernel_size",
            "t_kernel_size",
            "mlp_ratio",
            "ada_rank",
            "ada_alpha",
            "ada_mode",
            "act_type",
            "pe_type",
            "scale",
            "decode_type",
            "use_adanorm",
            "causal",
            "sampling_rate",
            "n_fft",
            "hop_size",
            "win_size",
            "fmin",
            "fmax",
            "num_mels",
            "harmonic_num",
            "sine_amp",
            "add_noise_std",
            "voiced_threshold",
        ] if k in hparams}
        self.dnn = NsfBcdBridge(**dnn_kwargs)
        missing, unexpected = self.dnn.load_state_dict(dnn_state_dict, strict=False)
        if missing or unexpected:
            print(f"[BridgeVoC] load_state_dict missing={len(missing)} unexpected={len(unexpected)}")

        self.eval()

    def _spec_fwd(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1.0:
                e = self.spec_abs_exponent
                spec = spec.abs() ** e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            pass
        return spec

    def _spec_back(self, spec: torch.Tensor) -> torch.Tensor:
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1.0:
                e = self.spec_abs_exponent
                spec = spec.abs() ** (1.0 / e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1.0) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            pass
        return spec


def load_bridgevoc_inference(
    ckpt_path: str,
    infer_steps: Optional[int] = None,
) -> BridgeVocInference:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {}) or {}
    state_dict = ckpt.get("state_dict", {}) or {}
    dnn_state = {k[len("dnn."):]: v for k, v in state_dict.items() if k.startswith("dnn.")}
    return BridgeVocInference(hparams, dnn_state, infer_steps=infer_steps)

