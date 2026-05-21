"""Activation capture hooks used by datastore builders."""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable

from torch import nn


class KeyType(Enum):
    """Which activation tensor to use as the kNN key."""

    last_ffn_input = auto()
    last_ffn_output = auto()

    @staticmethod
    def from_string(s: str) -> "KeyType":
        return KeyType[s.lower()]


class ActivationCapturer(nn.Module):
    """Forward-hook payload: stashes the captured tensor on `self.captured`."""

    def __init__(self, layer: nn.Module, capture_input: bool = False) -> None:
        super().__init__()
        self.layer = layer
        self.capture_input = capture_input
        self.captured = None

    def forward(self, module, input, output):  # noqa: A002 — torch hook signature
        self.captured = (input[0] if self.capture_input else output).detach()


# (model_type, key_type) -> (selector_lambda, capture_input)
MODEL_LAYER_TO_CAPTURE: dict[str, dict[KeyType, tuple[Callable[[nn.Module], nn.Module], bool]]] = {
    "gpt2": {
        KeyType.last_ffn_input: (lambda model: model.base_model.h[-1].mlp, True),
        KeyType.last_ffn_output: (lambda model: model.base_model.h[-1], False),
    },
    "bart": {
        KeyType.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
        KeyType.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
    },
    "marian": {
        KeyType.last_ffn_input: (lambda model: model.base_model.decoder.layers[-1].fc1, True),
        KeyType.last_ffn_output: (lambda model: model.base_model.decoder.layers[-1], False),
    },
    "t5": {
        KeyType.last_ffn_input: (
            lambda model: model.base_model.decoder.block[-1].layer[2].DenseReluDense,
            True,
        ),
        KeyType.last_ffn_output: (lambda model: model.base_model.decoder.block[-1].layer[2], False),
    },
}


def get_model_last_layer(model_type: str) -> Callable[[nn.Module], nn.Module]:
    return lambda model: model.lm_head


def get_model_embedding_layer(model_type: str) -> Callable[[nn.Module], nn.Module]:
    if model_type.startswith("gpt2"):
        return lambda model: model.transformer.wte
    raise NotImplementedError(f"No embedding-layer accessor for {model_type}")
