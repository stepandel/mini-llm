#!/usr/bin/env python3
"""
Fine-tune GPT-2 model for spam classification.

This script loads a pre-trained GPT-2 model, freezes most layers except the last
transformer block and output head, and fine-tunes it on a spam/ham dataset.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader

from modules.gpt_utils import create_balanced_dataset, load_weights_into_gpt, plot_values
from modules.gpt_fine_tune_classifier import (
    random_split, SpamDataset, GPTClassifierModel, 
    cal_accuracy_loader, calc_loss_loader, train_classifier_simple
)
from modules.gpt_model import GPTModel
from modules.gpt_download import download_and_load_gpt2


def generate_classifier_output_path(model_size, context_length, output_dir="checkpoints"):
    """Generate deterministic output path for classifier model based on config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"classifier_gpt{model_size}_ctx{context_length}_{timestamp}.pth"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path / filename


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 for spam classification"
    )
    parser.add_argument(
        "--data", 
        type=str, 
        default="SMSSpamCollection",
        help="Path to the SMS spam collection dataset"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt2-small (124M)",
        choices=["gpt2-small (124M)", "gpt2-medium (355M)", "gpt2-large (774M)", "gpt2-xl (1558M)"],
        help="GPT-2 model size to use"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=5,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--weight-decay", 
        type=float, 
        default=0.1,
        help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=128,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="checkpoints",
        help="Output directory for saved models (filename auto-generated)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=123,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--train-split", 
        type=float, 
        default=0.7,
        help="Proportion of data for training"
    )
    parser.add_argument(
        "--val-split", 
        type=float, 
        default=0.1,
        help="Proportion of data for validation"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the fine-tuning pipeline."""
    args = parse_args()
    
    print("=" * 80)
    print("GPT-2 Spam Classifier Fine-tuning")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Max sequence length: {args.max_length}")
    print(f"Random seed: {args.seed}")
    print("=" * 80)
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Load and prepare dataset
    print("\n[1/8] Loading and preparing dataset...")
    df = pd.read_csv(args.data, sep='\t', header=None, names=['Label', 'Text'])
    balanced_df = create_balanced_dataset(df)
    balanced_df["Label"] = balanced_df["Label"].map({'ham': 0, 'spam': 1})
    
    # Split dataset
    print("[2/8] Splitting dataset...")
    train_df, validation_df, test_df = random_split(
        balanced_df, args.train_split, args.val_split
    )
    
    # Save splits
    train_df.to_csv('train.csv', index=None)
    validation_df.to_csv('validation.csv', index=None)
    test_df.to_csv('test.csv', index=None)
    print(f"  Train: {len(train_df)} samples")
    print(f"  Validation: {len(validation_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Create datasets and dataloaders
    print("[3/8] Creating datasets and dataloaders...")
    tokenizer = tiktoken.get_encoding("gpt2")
    
    train_dataset = SpamDataset("train.csv", tokenizer, max_length=args.max_length)
    val_dataset = SpamDataset("validation.csv", tokenizer, max_length=train_dataset.max_length)
    test_dataset = SpamDataset("test.csv", tokenizer, max_length=train_dataset.max_length)
    
    num_workers = 0
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        drop_last=True
    )
    
    # Configure model
    print("[4/8] Configuring and loading model...")
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    BASE_CONFIG.update(model_configs[args.model])
    
    assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
        f"`max_length={BASE_CONFIG['context_length']}`"
    )
    
    # Download and load GPT-2 weights
    model_size = args.model.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    
    # Initialize model
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    
    # Freeze all parameters initially
    for param in model.parameters():
        param.requires_grad = False
    
    # Set the output head to binary classification head
    model = GPTClassifierModel(model, num_labels=2)
    
    # Make the last transformer block and final norm layer trainable
    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True
    for param in model.final_norm.parameters():
        param.requires_grad = True
    
    # Move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Using device: {device}")
    model.to(device)
    
    # Calculate initial metrics
    print("[5/8] Calculating initial metrics...")
    torch.manual_seed(args.seed)
    
    train_accuracy = cal_accuracy_loader(train_loader, model, device, num_batches=10)
    val_accuracy = cal_accuracy_loader(val_loader, model, device, num_batches=10)
    test_accuracy = cal_accuracy_loader(test_loader, model, device, num_batches=10)
    
    print(f"  Initial train accuracy: {train_accuracy:.2%}")
    print(f"  Initial val accuracy: {val_accuracy:.2%}")
    print(f"  Initial test accuracy: {test_accuracy:.2%}")
    
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=10)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=10)
        test_loss = calc_loss_loader(test_loader, model, device, num_batches=10)
    
    print(f"  Initial train loss: {train_loss:.4f}")
    print(f"  Initial val loss: {val_loss:.4f}")
    print(f"  Initial test loss: {test_loss:.4f}")
    
    # Training
    print("[6/8] Training model...")
    start_time = time.time()
    torch.manual_seed(args.seed)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device, args.epochs, 50, 5
    )
    
    end_time = time.time()
    training_time = (end_time - start_time) / 60
    print(f"  Training completed in {training_time:.2f} minutes")
    
    # Final evaluation
    print("[7/8] Final evaluation...")
    final_train_accuracy = cal_accuracy_loader(train_loader, model, device)
    final_val_accuracy = cal_accuracy_loader(val_loader, model, device)
    final_test_accuracy = cal_accuracy_loader(test_loader, model, device)
    
    print(f"  Final train accuracy: {final_train_accuracy:.2%}")
    print(f"  Final val accuracy: {final_val_accuracy:.2%}")
    print(f"  Final test accuracy: {final_test_accuracy:.2%}")
    
    # Generate output path with model config and timestamp
    output_path = generate_classifier_output_path(
        model_size=model_size,
        context_length=BASE_CONFIG["context_length"],
        output_dir=args.output_dir
    )
    
    # Save model
    print(f"[8/8] Saving model to {output_path}...")
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": BASE_CONFIG,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_accs": train_accs,
        "val_accs": val_accs,
        "final_train_accuracy": final_train_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "final_test_accuracy": final_test_accuracy,
        "training_args": vars(args),
        "timestamp": datetime.now().isoformat()
    }, output_path)
    print(f"  Model saved successfully to {output_path}")
    
    # Plot training curves
    print("Generating training plots...")
    epochs_tensor = torch.linspace(0, args.epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)
    
    # Print summary
    print("\n" + "=" * 80)
    print("Training Summary")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Training time: {training_time:.2f} minutes")
    print(f"Final train accuracy: {final_train_accuracy:.2%}")
    print(f"Final val accuracy: {final_val_accuracy:.2%}")
    print(f"Final test accuracy: {final_test_accuracy:.2%}")
    print(f"Model saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()