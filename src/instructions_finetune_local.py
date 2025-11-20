#!/usr/bin/env python3
"""
Instruction Fine-tuning Script for GPT Models

This script fine-tunes a GPT model on instruction-following tasks.
It loads a pre-trained GPT-2 model, fine-tunes it on instruction data,
evaluates the model, and saves the results.
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from functools import partial
from pathlib import Path

import tiktoken
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.gpt_fine_tune_instructions import custom_collate_fn, InstructionDataset, format_input
from ollama_evaluator import generate_model_scores
from modules.gpt_utils import download_and_load_file, download_and_load_gpt2, load_weights_into_gpt
from modules.gpt_model import GPTModel, generate
from modules.gpt_pretraining import train_model_simple
from modules.gpt_utils import plot_losses
from modules.gpt_data import text_to_token_ids, token_ids_to_text


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('instruction_finetuning.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def generate_instruction_output_path(model_size, context_length, output_dir="checkpoints"):
    """Generate deterministic output path for instruction-tuned model based on config and timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"instruction_gpt{model_size}_ctx{context_length}_{timestamp}.pth"
    
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path / filename


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Fine-tune GPT model on instruction-following tasks'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2-medium (355M)',
        choices=['gpt2-small (124M)', 'gpt2-medium (355M)', 'gpt2-large (774M)', 'gpt2-xl (1558M)'],
        help='GPT model size to use'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size for training (default: 8)'
    )
    
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=2,
        help='Number of training epochs (default: 2)'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.00005,
        help='Learning rate (default: 0.00005)'
    )
    
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.1,
        help='Weight decay (default: 0.1)'
    )
    
    parser.add_argument(
        '--eval-freq',
        type=int,
        default=5,
        help='Evaluation frequency during training (default: 5)'
    )
    
    parser.add_argument(
        '--eval-iter',
        type=int,
        default=5,
        help='Number of evaluation iterations (default: 5)'
    )
    
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum new tokens to generate (default: 256)'
    )
    
    parser.add_argument(
        '--data-url',
        type=str,
        default='https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json',
        help='URL to download instruction data from'
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        default='instruction-data.json',
        help='Path to instruction data file (default: instruction-data.json)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Directory to save outputs (default: current directory)'
    )
    
    parser.add_argument(
        '--models-dir',
        type=str,
        default='gpt2',
        help='Directory containing GPT-2 model files (default: gpt2)'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        help='Number of workers for data loading (default: 0)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=123,
        help='Random seed for reproducibility (default: 123)'
    )
    
    parser.add_argument(
        '--skip-evaluation',
        action='store_true',
        help='Skip model evaluation with ollama'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Skip plotting losses'
    )
    
    return parser.parse_args()


def setup_device():
    """Setup and return the appropriate device for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    # Note: MPS is available on macOS but is not stable for training
    # Uncomment if you want to use MPS anyway:
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     logger.info("Using MPS device")
    
    return device


def load_and_split_data(file_path, url, train_ratio=0.85, test_ratio=0.1):
    """Load data and split into train/val/test sets."""
    logger.info(f"Loading data from {file_path}")
    data = download_and_load_file(file_path, url)
    logger.info(f"Loaded {len(data)} instruction examples")
    
    train_portion = int(len(data) * train_ratio)
    test_portion = int(len(data) * test_ratio)
    
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    logger.info(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def create_data_loaders(train_data, val_data, test_data, tokenizer, batch_size, num_workers, device):
    """Create data loaders for training, validation, and testing."""
    customized_collate_fn = partial(
        custom_collate_fn,
        ignore_index=-100,
        device=device
    )
    
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )
    
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )
    
    return train_loader, val_loader, test_loader


def load_model(model_name, models_dir):
    """Load and configure GPT model."""
    BASE_CONFIG = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }
    
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }
    
    logger.info(f"Loading model: {model_name}")
    BASE_CONFIG.update(model_configs[model_name])
    
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir=models_dir
    )
    
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    
    logger.info(f"Model loaded successfully with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, BASE_CONFIG


def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, val_data, tokenizer):
    """Train the model."""
    logger.info("Starting model training...")
    start_time = time.time()
    
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    logger.info(f"Training completed in {execution_time_minutes:.2f} minutes.")
    
    return train_losses, val_losses, tokens_seen, execution_time_minutes


def generate_test_responses(model, test_data, tokenizer, device, config, max_new_tokens):
    """Generate model responses for test data."""
    logger.info("Generating responses for test data...")
    
    for i, entry in tqdm(enumerate(test_data), total=len(test_data), desc="Generating responses"):
        input_text = format_input(entry)
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=max_new_tokens,
            context_size=config["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        
        response_text = generated_text[len(input_text):].replace("### Response:\n", "").strip()
        test_data[i]["model_response"] = response_text
    
    return test_data


def save_results(model, optimizer, test_data, model_name, config, train_losses, val_losses, 
                 tokens_seen, training_time, training_args, output_dir):
    """Save model and test results with comprehensive metadata."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save test data with model responses
    response_file = os.path.join(output_dir, "instruction-data-with-model-response.json")
    with open(response_file, "w") as f:
        json.dump(test_data, f, indent=4)
    logger.info(f"Test data with responses saved to {response_file}")
    
    # Generate output path with model config and timestamp
    model_size = model_name.split(" ")[-1].lstrip("(").rstrip(")")
    model_path = generate_instruction_output_path(
        model_size=model_size,
        context_length=config["context_length"],
        output_dir=output_dir
    )
    
    # Save model with comprehensive metadata
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "tokens_seen": tokens_seen,
        "training_time_minutes": training_time,
        "training_args": training_args,
        "timestamp": datetime.now().isoformat()
    }, model_path)
    logger.info(f"Model with metadata saved to {model_path}")
    
    return response_file, model_path


def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        logger.info("=" * 80)
        logger.info("Starting Instruction Fine-tuning")
        logger.info("=" * 80)
        logger.info(f"Configuration: {vars(args)}")
        
        # Set random seed for reproducibility
        torch.manual_seed(args.seed)
        logger.info(f"Random seed set to {args.seed}")
        
        # Setup device
        device = setup_device()
        
        # Load and split data
        train_data, val_data, test_data = load_and_split_data(
            args.data_file, args.data_url
        )
        
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        logger.info("Tokenizer initialized")
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            train_data, val_data, test_data, tokenizer, 
            args.batch_size, args.num_workers, device
        )
        
        # Load model
        model, config = load_model(args.model, args.models_dir)
        model = model.to(device)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        logger.info(f"Optimizer: AdamW (lr={args.learning_rate}, weight_decay={args.weight_decay})")
        
        # Train model
        train_losses, val_losses, tokens_seen, training_time = train_model(
            model, train_loader, val_loader, optimizer, device,
            args.num_epochs, args.eval_freq, args.eval_iter, val_data, tokenizer
        )
        
        # Plot losses
        if not args.no_plot:
            logger.info("Plotting training losses...")
            epochs_tensor = torch.linspace(1, args.num_epochs, len(train_losses))
            plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
        
        # Generate test responses
        test_data = generate_test_responses(
            model, test_data, tokenizer, device, config, args.max_new_tokens
        )
        
        # Save results
        response_file, model_path = save_results(
            model=model,
            optimizer=optimizer,
            test_data=test_data,
            model_name=args.model,
            config=config,
            train_losses=train_losses,
            val_losses=val_losses,
            tokens_seen=tokens_seen,
            training_time=training_time,
            training_args=vars(args),
            output_dir=args.output_dir
        )
        
        # Evaluate with ollama
        if not args.skip_evaluation:
            logger.info("Evaluating model responses with ollama...")
            try:
                scores = generate_model_scores(test_data, "model_response")
                logger.info(f"Evaluation scores: {scores}")
            except Exception as e:
                logger.warning(f"Could not evaluate with ollama: {e}")
        
        logger.info("=" * 80)
        logger.info("Fine-tuning completed successfully!")
        logger.info(f"Training time: {training_time:.2f} minutes")
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Test responses saved: {response_file}")
        logger.info("=" * 80)
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())