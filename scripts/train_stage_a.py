#!/usr/bin/env python3
"""
Stage A: Arrangement Generator Training Script
Transformer-XL for music arrangement generation from vocal melody

Usage:
    python scripts/train_stage_a.py \
        --data-dir ./data/processed \
        --batch-size 32 \
        --epochs 50 \
        --device tpu
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ArrangementGenerator(nn.Module):
    """Simplified Transformer-XL for music arrangement generation"""
    
    def __init__(
        self,
        vocab_size: int = 512,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 4096,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_length, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Conditioning embeddings
        self.genre_embedding = nn.Embedding(8, d_model)
        self.tala_embedding = nn.Embedding(16, d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
    
    def forward(
        self,
        token_ids: torch.Tensor,
        genre_ids: Optional[torch.Tensor] = None,
        tala_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = token_ids.shape
        
        x = self.token_embedding(token_ids)
        
        positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        
        if genre_ids is not None:
            genre_emb = self.genre_embedding(genre_ids).unsqueeze(1)
            x = x + genre_emb
        
        if tala_ids is not None:
            tala_emb = self.tala_embedding(tala_ids).unsqueeze(1)
            x = x + tala_emb
        
        x = self.transformer(x)
        logits = self.output_projection(x)
        return logits

def create_dummy_dataset(num_samples: int = 100, seq_len: int = 256, batch_size: int = 32):
    """Create dummy dataset for testing"""
    logger.info(f"📊 Creating dummy dataset: {num_samples} samples")
    
    token_ids = torch.randint(0, 512, (num_samples, seq_len))
    genre_ids = torch.randint(0, 4, (num_samples,))
    tala_ids = torch.randint(0, 8, (num_samples,))
    labels = torch.randint(0, 512, (num_samples, seq_len))
    
    dataset = TensorDataset(token_ids, genre_ids, tala_ids, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    epoch: int
) -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        token_ids, genre_ids, tala_ids, labels = batch
        token_ids = token_ids.to(device)
        genre_ids = genre_ids.to(device)
        tala_ids = tala_ids.to(device)
        labels = labels.to(device)
        
        logits = model(token_ids, genre_ids, tala_ids)
        
        batch_size, seq_len, vocab_size = logits.shape
        loss = criterion(logits.view(-1, vocab_size), labels.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if WANDB_AVAILABLE and batch_idx % 10 == 0:
            wandb.log({'train_loss': loss.item(), 'batch': batch_idx})
    
    avg_loss = total_loss / len(dataloader)
    logger.info(f"📊 Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    
    return avg_loss

def main():
    parser = argparse.ArgumentParser(description="Train Stage A - Arrangement Generator")
    parser.add_argument('--data-dir', type=str, default='./data/processed', help='Data directory')
    parser.add_argument('--output-dir', type=str, default='./outputs/checkpoints/stage_a', help='Output directory')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu', 'tpu'], help='Device')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases')
    
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info("=" * 70)
    logger.info("🎼 Stage A: Arrangement Generator Training")
    logger.info("=" * 70)
    logger.info(f"📊 Config: Batch Size={args.batch_size}, Epochs={args.epochs}, LR={args.learning_rate}")
    
    device = 'cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu'
    logger.info(f"🖥️  Using device: {device}")
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project="tamil-music-model-4",
            name="stage_a_arrangement_generator",
            config=vars(args)
        )
    
    logger.info("📦 Building model...")
    model = ArrangementGenerator(
        vocab_size=512,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📊 Model Parameters: {total_params / 1e6:.2f}M")
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * 100, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    logger.info("📥 Loading dataset...")
    dataloader = create_dummy_dataset(num_samples=1000, batch_size=args.batch_size)
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("🚀 Starting training...")
    best_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        avg_loss = train_epoch(model, dataloader, optimizer, criterion, device, epoch)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = Path(args.output_dir) / 'best.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"✅ Best checkpoint saved: {checkpoint_path}")
        
        if epoch % 10 == 0:
            checkpoint_path = Path(args.output_dir) / f'epoch_{epoch}.pt'
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info(f"✅ Training complete! Best loss: {best_loss:.4f}")
    logger.info("=" * 70)
    
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()

if __name__ == "__main__":
    main()
