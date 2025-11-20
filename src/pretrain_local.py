#!/usr/bin/env python3
"""
Local GPT Model Pretraining Script

This script provides a command-line interface for pretraining a GPT model
on local text data with configurable hyperparameters. Output files are
automatically named with model configuration and timestamp.

Usage:
    python pretrain_local.py --data the-verdict.txt --epochs 10 --batch-size 2
    
Output naming format:
    checkpoints/gpt124M_ctx256_20231119_143022.pth
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import tiktoken
from gpt_model import GPTModel
from gpt_data import create_dataloader
from gpt_pretraining import train_model_simple


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)


def get_gpt_config(context_length=256):
    """Get GPT-124M configuration with customizable context length."""
    return {
        "vocab_size": 50257,
        "context_length": context_length,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.1,
        "qkv_bias": False
    }


def get_model_name(config):
    """Generate a model name based on configuration."""
    emb_dim = config["emb_dim"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    context_length = config["context_length"]
    
    # Determine model size based on embedding dimension
    if emb_dim == 768 and n_layers == 12:
        size = "124M"
    elif emb_dim == 1024 and n_layers == 24:
        size = "355M"
    elif emb_dim == 1280 and n_layers == 36:
        size = "774M"
    elif emb_dim == 1600 and n_layers == 48:
        size = "1558M"
    else:
        # Custom configuration
        size = f"{emb_dim}d{n_layers}l"
    
    return f"gpt{size}_ctx{context_length}"


def generate_output_path(config, output_dir="checkpoints"):
    """Generate deterministic output path based on config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = get_model_name(config)
    filename = f"{model_name}_{timestamp}.pth"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path / filename


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Pretrain GPT model on local text data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data arguments
    parser.add_argument(
        '--data', 
        type=str, 
        default='the-verdict.txt',
        help='Path to training text file'
    )
    parser.add_argument(
        '--train-ratio', 
        type=float, 
        default=0.90,
        help='Ratio of data to use for training (rest for validation)'
    )
    
    # Model arguments
    parser.add_argument(
        '--context-length', 
        type=int, 
        default=256,
        help='Context length for the model'
    )
    
    # Training arguments
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=2,
        help='Batch size for training'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.0004,
        help='Learning rate'
    )
    parser.add_argument(
        '--weight-decay', 
        type=float, 
        default=0.1,
        help='Weight decay for AdamW optimizer'
    )
    parser.add_argument(
        '--eval-freq', 
        type=int, 
        default=5,
        help='Evaluate every N batches'
    )
    parser.add_argument(
        '--eval-iter', 
        type=int, 
        default=5,
        help='Number of iterations for evaluation'
    )
    parser.add_argument(
        '--start-context', 
        type=str, 
        default='Every effort moves you',
        help='Starting context for text generation during training'
    )
    
    # Output arguments
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='checkpoints',
        help='Output directory for saved models (filename auto-generated)'
    )
    parser.add_argument(
        '--seed', 
        type=int, 
        default=123,
        help='Random seed for reproducibility'
    )
    
    return parser.parse_args()


def load_text_data(file_path):
    """Load text data from file."""
    logger.info(f"Loading data from {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()
        logger.info(f"Loaded {len(text_data):,} characters")
        return text_data
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        sys.exit(1)


def main():
    """Main training function."""
    args = parse_arguments()
    
    # Set random seed
    torch.manual_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Load data
    text_data = load_text_data(args.data)
    
    # Split data
    split_idx = int(args.train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    logger.info(f"Split data: {len(train_data):,} chars for training, {len(val_data):,} chars for validation")
    
    # Get model configuration
    gpt_config = get_gpt_config(context_length=args.context_length)
    logger.info(f"Model configuration: {gpt_config}")
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader = create_dataloader(
        train_data,
        batch_size=args.batch_size,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = create_dataloader(
        val_data,
        batch_size=args.batch_size,
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = GPTModel(gpt_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize tokenizer and optimizer
    tokenizer = tiktoken.get_encoding("gpt2")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    logger.info(f"Optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})")
    
    # Train model
    logger.info(f"Starting training for {args.epochs} epochs...")
    start_time = time.time()
    
    try:
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=args.epochs, 
            eval_freq=args.eval_freq, 
            eval_iter=args.eval_iter,
            start_context=args.start_context, 
            tokenizer=tokenizer
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        sys.exit(1)
    
    training_time = time.time() - start_time
    logger.info(f"Training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    logger.info(f"Tokens processed: {tokens_seen[-1]:,}")
    
    # Generate output path with model config and timestamp
    output_path = generate_output_path(gpt_config, output_dir=args.output_dir)
    
    # Save model
    logger.info(f"Saving model to {output_path}")
    try:
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": gpt_config,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "tokens_seen": tokens_seen,
            "training_args": vars(args),
            "timestamp": datetime.now().isoformat()
        }, output_path)
        logger.info("Model saved successfully")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        sys.exit(1)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"Final training loss: {train_losses[-1]:.4f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
    logger.info(f"Total training time: {training_time/60:.2f} minutes")
    logger.info(f"Tokens processed: {tokens_seen[-1]:,}")
    logger.info(f"Model saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()