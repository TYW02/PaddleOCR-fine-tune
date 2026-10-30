"""Small PyTorch line recogniser (CRNN + CTC) used as a FROZEN loss for super-resolution (M2).

Why a recogniser inside the SR loss: every text-SR method that improves reading ties its loss to a
text recogniser (TextSR's text perceptual loss, TPGSR / TATT text priors, Text Gestalt's stroke
attention). A pixel loss rewards the average of the possible letters (a soft blob); a recognition loss
punishes a blob that reads as the wrong letter. PaddleOCR cannot be used for this (Paddle, not
PyTorch: no gradients), so this small CRNN is trained on IAM handwritten lines plus our own labelled
whiteboard crops, then frozen. The SR model is still JUDGED by PaddleOCR, a different recogniser, so a
gain cannot come from fooling this one.

Input convention: the same normalised grey space the SR model works in (pair_data.normalise_board,
board ~1.0, ink darker), turned into INK = 1 - grey so that padding with 0 means "empty board".
Height HEIGHT px, width scaled with the aspect ratio (capped at MAX_WIDTH).
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

HEIGHT = 64
MAX_WIDTH = 1024
DOWNSAMPLE = 4          # width reduction of the CNN: one CTC time step per 4 input columns


# ------------------------------------------------------------------------------------------------
# Characters
# ------------------------------------------------------------------------------------------------
class Charset:
    """Index 0 is the CTC blank; characters are 1..N."""

    def __init__(self, chars: str):
        self.chars = "".join(sorted(set(chars)))
        self.index = {c: i + 1 for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars) + 1

    def encode(self, text: str) -> list[int]:
        """Unknown characters are dropped (reported by the training script)."""
        return [self.index[c] for c in text if c in self.index]

    def decode_greedy(self, ids) -> str:
        out, prev = [], 0
        for i in ids:
            i = int(i)
            if i != prev and i != 0:
                out.append(self.chars[i - 1])
            prev = i
        return "".join(out)


def normalise_text(text: str) -> str:
    """Same text normalisation as the notebook's CER (lower case, single spaces)."""
    return " ".join(str(text).lower().split())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, prediction: str) -> float:
    reference, prediction = normalise_text(reference), normalise_text(prediction)
    if not reference:
        return 0.0 if not prediction else 1.0
    return min(1.0, edit_distance(reference, prediction) / len(reference))


# ------------------------------------------------------------------------------------------------
# Preprocessing (NumPy, for training data) and its differentiable twin (torch, for SR outputs)
# ------------------------------------------------------------------------------------------------
def _prefactor(h: int) -> int:
    """Integer pre-shrink for tall crops: average k x k blocks first (no aliasing of thin strokes), then
    a plain bilinear resize to HEIGHT. The NumPy and torch paths use exactly the same two steps, so the
    recogniser sees the same kind of image in training and when it scores SR outputs."""
    return max(1, h // (2 * HEIGHT))


def ink_from_norm_grey(grey01: np.ndarray) -> np.ndarray:
    """Normalised grey (H, W) in [0,1] -> INK image (HEIGHT, w) float32 in [0,1]."""
    grey01 = grey01.astype(np.float32)
    h, w = grey01.shape[:2]
    k = _prefactor(h)
    if k > 1:
        grey01 = grey01[:h // k * k, :w // k * k].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    new_w = int(np.clip(round(w * HEIGHT / h), 8, MAX_WIDTH))
    small = cv2.resize(grey01, (new_w, HEIGHT), interpolation=cv2.INTER_LINEAR)
    return np.clip(1.0 - small, 0.0, 1.0)


def ink_from_rgb(rgb_uint8: np.ndarray) -> np.ndarray:
    """A photo crop (uint8 RGB or grey) -> INK image, through the SR model's own normalisation."""
    import pair_data                                  # local import: keeps this module importable alone
    if rgb_uint8.ndim == 2:
        rgb_uint8 = cv2.cvtColor(rgb_uint8, cv2.COLOR_GRAY2RGB)
    grey = pair_data.normalise_board(rgb_uint8).mean(axis=2)
    return ink_from_norm_grey(grey)


def ink_from_sr_output(norm_grey: torch.Tensor) -> torch.Tensor:
    """Differentiable twin of ink_from_norm_grey for SR outputs: (B, C, H, W) normalised grey in [0,1]
    -> (B, 1, HEIGHT, w) INK. Gradients flow back into the SR model."""
    _, _, h, w = norm_grey.shape
    x = norm_grey[:, :1]
    k = _prefactor(h)
    if k > 1:
        x = F.avg_pool2d(x, k)                        # same k x k block average as the NumPy path
    new_w = int(min(max(round(w * HEIGHT / h), 8), MAX_WIDTH))
    small = F.interpolate(x, size=(HEIGHT, new_w), mode="bilinear", align_corners=False)
    return (1.0 - small).clamp(0.0, 1.0)


def pad_batch(inks: list[np.ndarray]) -> tuple[torch.Tensor, torch.Tensor]:
    """List of (HEIGHT, w) INK arrays -> (B, 1, HEIGHT, W_max) tensor padded with 0 (empty board) and
    the CTC input lengths (time steps that cover real columns)."""
    w_max = max(i.shape[1] for i in inks)
    w_max = int(np.ceil(w_max / DOWNSAMPLE) * DOWNSAMPLE)
    batch = np.zeros((len(inks), 1, HEIGHT, w_max), np.float32)
    for k, ink in enumerate(inks):
        batch[k, 0, :, :ink.shape[1]] = ink
    lengths = torch.tensor([max(1, i.shape[1] // DOWNSAMPLE) for i in inks], dtype=torch.long)
    return torch.from_numpy(batch), lengths


# ------------------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------------------
def _block(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True)]


class LineCRNN(nn.Module):
    """CNN (height 64 -> 4, width / 4) -> max over the remaining height (column pooling, as in
    'Best practices for a handwritten text recognition system') -> 2-layer BLSTM -> per-column classes.
    About 4 M parameters."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = nn.Sequential(
            *_block(1, 32), nn.MaxPool2d(2, 2),                     # 32 x W/2
            *_block(32, 64), nn.MaxPool2d(2, 2),                    # 16 x W/4
            *_block(64, 128), *_block(128, 128), nn.MaxPool2d((2, 1)),   # 8 x W/4
            *_block(128, 256), *_block(256, 256), nn.MaxPool2d((2, 1)),  # 4 x W/4
            nn.Dropout2d(0.1),
        )
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
        self.head = nn.Linear(512, n_classes)

    def forward(self, ink: torch.Tensor) -> torch.Tensor:
        """(B, 1, HEIGHT, W) -> log-probabilities (T, B, n_classes), T = W / 4 (the layout F.ctc_loss wants)."""
        features = self.cnn(ink).amax(dim=2)             # (B, 256, T)
        sequence, _ = self.rnn(features.transpose(1, 2))  # (B, T, 512)
        return self.head(sequence).log_softmax(dim=2).transpose(0, 1)


def ctc_loss(log_probs: torch.Tensor, input_lengths: torch.Tensor, labels: list[list[int]]) -> torch.Tensor:
    """Mean CTC loss per character. Computed on the CPU (CTC is not implemented on every backend, e.g.
    Apple MPS); the .cpu() copy is differentiable, so gradients still reach the model on its device."""
    targets = torch.tensor([c for lab in labels for c in lab], dtype=torch.long)
    target_lengths = torch.tensor([len(lab) for lab in labels], dtype=torch.long)
    return F.ctc_loss(log_probs.float().cpu(), targets, input_lengths.cpu(), target_lengths,
                      blank=0, reduction="mean", zero_infinity=True)


@torch.no_grad()
def read(model: LineCRNN, charset: Charset, ink_batch: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    log_probs = model(ink_batch)
    best = log_probs.argmax(dim=2).transpose(0, 1).cpu()      # (B, T)
    return [charset.decode_greedy(best[k, :int(lengths[k])]) for k in range(best.shape[0])]


def save(path, model: LineCRNN, charset: Charset, **info) -> None:
    torch.save({"model": model.state_dict(), "chars": charset.chars, "height": HEIGHT,
                "downsample": DOWNSAMPLE, **info}, path)


def load(path, device) -> tuple[LineCRNN, Charset, dict]:
    """Load a trained recogniser, FROZEN (eval mode, no parameter gradients)."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("height") != HEIGHT or checkpoint.get("downsample") != DOWNSAMPLE:
        raise ValueError(f"{path} was trained for a different input size")
    charset = Charset(checkpoint["chars"])
    model = LineCRNN(len(charset))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, charset, checkpoint
