from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0, dtype=np.float32)


def _linear(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    return x @ weight.T + bias


def _batch_norm_eval(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray,
    running_mean: np.ndarray,
    running_var: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    return ((x - running_mean) / np.sqrt(running_var + eps)) * weight + bias


@dataclass
class ResidualBlockWeights:
    norm1_weight: np.ndarray
    norm1_bias: np.ndarray
    norm1_running_mean: np.ndarray
    norm1_running_var: np.ndarray
    linear1_weight: np.ndarray
    linear1_bias: np.ndarray
    norm2_weight: np.ndarray
    norm2_bias: np.ndarray
    norm2_running_mean: np.ndarray
    norm2_running_var: np.ndarray
    linear2_weight: np.ndarray
    linear2_bias: np.ndarray


class NumpyTabularResNet:
    def __init__(self, weights_path: Path) -> None:
        arrays = np.load(weights_path)
        self.input_weight = arrays["input_weight"].astype(np.float32)
        self.input_bias = arrays["input_bias"].astype(np.float32)
        self.head_norm_weight = arrays["head_norm_weight"].astype(np.float32)
        self.head_norm_bias = arrays["head_norm_bias"].astype(np.float32)
        self.head_norm_running_mean = arrays["head_norm_running_mean"].astype(np.float32)
        self.head_norm_running_var = arrays["head_norm_running_var"].astype(np.float32)
        self.head_linear_weight = arrays["head_linear_weight"].astype(np.float32)
        self.head_linear_bias = arrays["head_linear_bias"].astype(np.float32)
        self.depth = int(arrays["meta_depth"][0])
        self.blocks = []
        for idx in range(self.depth):
            self.blocks.append(
                ResidualBlockWeights(
                    norm1_weight=arrays[f"blocks_{idx}_norm1_weight"].astype(np.float32),
                    norm1_bias=arrays[f"blocks_{idx}_norm1_bias"].astype(np.float32),
                    norm1_running_mean=arrays[f"blocks_{idx}_norm1_running_mean"].astype(np.float32),
                    norm1_running_var=arrays[f"blocks_{idx}_norm1_running_var"].astype(np.float32),
                    linear1_weight=arrays[f"blocks_{idx}_linear1_weight"].astype(np.float32),
                    linear1_bias=arrays[f"blocks_{idx}_linear1_bias"].astype(np.float32),
                    norm2_weight=arrays[f"blocks_{idx}_norm2_weight"].astype(np.float32),
                    norm2_bias=arrays[f"blocks_{idx}_norm2_bias"].astype(np.float32),
                    norm2_running_mean=arrays[f"blocks_{idx}_norm2_running_mean"].astype(np.float32),
                    norm2_running_var=arrays[f"blocks_{idx}_norm2_running_var"].astype(np.float32),
                    linear2_weight=arrays[f"blocks_{idx}_linear2_weight"].astype(np.float32),
                    linear2_bias=arrays[f"blocks_{idx}_linear2_bias"].astype(np.float32),
                )
            )

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = _linear(x.astype(np.float32), self.input_weight, self.input_bias)
        for block in self.blocks:
            residual = out
            out = _batch_norm_eval(
                out,
                block.norm1_weight,
                block.norm1_bias,
                block.norm1_running_mean,
                block.norm1_running_var,
            )
            out = _relu(out)
            out = _linear(out, block.linear1_weight, block.linear1_bias)
            out = _batch_norm_eval(
                out,
                block.norm2_weight,
                block.norm2_bias,
                block.norm2_running_mean,
                block.norm2_running_var,
            )
            out = _relu(out)
            out = _linear(out, block.linear2_weight, block.linear2_bias)
            out = residual + out
        out = _batch_norm_eval(
            out,
            self.head_norm_weight,
            self.head_norm_bias,
            self.head_norm_running_mean,
            self.head_norm_running_var,
        )
        out = _relu(out)
        return _linear(out, self.head_linear_weight, self.head_linear_bias)
