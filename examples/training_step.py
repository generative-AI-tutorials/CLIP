"""Minimal example for training CLIP with the standard contrastive loss.

This script shows how to build the symmetric cross-entropy loss used when
training CLIP and how to run a single toy epoch with randomly generated
image/text pairs. Replace the ``DummyImageTextDataset`` with your real
image-text data to train on actual content.
"""
from __future__ import annotations

import random
from typing import Tuple
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import clip


class DummyImageTextDataset(Dataset):
    """Creates pseudo image-text data for demonstrating the training step."""

    def __init__(self, num_samples: int, preprocess: nn.Module, image_size: int = 224) -> None:
        self.num_samples = num_samples
        self.preprocess = preprocess
        self.image_size = image_size
        self.templates = [
            "a photo of a dog",
            "a photo of a cat",
            "a photo of a car",
            "a photo of food",
            "a photo of a plant",
            "a photo of a building",
        ]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        image = self._make_image()
        caption = random.choice(self.templates)
        return self.preprocess(image), caption

    def _make_image(self) -> Image.Image:
        array = torch.randint(0, 256, (self.image_size, self.image_size, 3), dtype=torch.uint8).numpy()
        return Image.fromarray(array)


def contrastive_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    """Symmetric cross-entropy over image→text and text→image logits."""

    batch_size = logits_per_image.size(0)
    ground_truth = torch.arange(batch_size, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, ground_truth)
    loss_t = F.cross_entropy(logits_per_text, ground_truth)
    return (loss_i + loss_t) / 2.0


def run_training_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    for images, raw_texts in dataloader:
        images = images.to(device)
        text_tokens = clip.tokenize(raw_texts, truncate=True).to(device)

        optimizer.zero_grad()
        logits_per_image, logits_per_text = model(images, text_tokens)
        loss = contrastive_loss(logits_per_image, logits_per_text)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

    dataset = DummyImageTextDataset(num_samples=64, preprocess=preprocess)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6)

    epoch_loss = run_training_epoch(model, dataloader, optimizer, device)
    print(f"Finished epoch with loss: {epoch_loss:.4f}")


if __name__ == "__main__":
    main()
