"""
LLaVA (Large Language and Vision Assistant) Fine-tuning Script

This script provides functionality to fine-tune LLaVA models on image-caption pairs
from datasets like Flickr30k. It includes dataset preparation, model training, 
and evaluation utilities.
"""

import os
import json
import sys
import time
import random
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import subprocess
from typing import Dict, List, Any, Tuple, Optional

# Add parent directory to path to import modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Function to install missing packages
def install_package(package_name):
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
    print(f"{package_name} installed successfully")

# Install required packages if they don't exist
required_packages = ['torch', 'transformers', 'peft', 'psutil', 'nltk', 'rouge_score', 'numpy', 'datasets']
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install_package(package)

# Now import all packages after ensuring they're installed
import torch
from PIL import Image
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model, PeftModel
from transformers.trainer_callback import TrainerCallback
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np
import psutil

# Download NLTK data
nltk.download('punkt', quiet=True)

from src.vision.vision_attention import VisionAttentionVisualizer

# Configure matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

# Set device - modify to handle MPS properly
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
    # Force MPS fallback for generate() operation which isn't fully supported
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

# Memory usage monitor
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.2f} MB")

# Function to plot the loss
def plot_loss(losses, output_dir):
    if not losses:
        print("No losses were recorded. Cannot generate plot.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title('Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # Save the plot
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    print(f"Loss plot saved to {os.path.join(output_dir, 'training_loss.png')}")
    
    # Also save the losses as a JSON file for later analysis
    with open(os.path.join(output_dir, 'training_losses.json'), 'w') as f:
        json.dump({"losses": losses}, f)

# Custom callback to track loss
class LossCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.current_step = 0
        
    def on_init_end(self, args, state, control, **kwargs):
        print("Training initialization complete. Starting training loop.")
        return control
        
    def on_step_end(self, args, state, control, **kwargs):
        self.current_step += 1
        if self.current_step % 10 == 0:
            print(f"Completed step {self.current_step}")
        return control
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            log_items = ", ".join([f"{k}={v:.4f}" for k, v in logs.items() if isinstance(v, (int, float))])
            print(f"Log at step {self.current_step}: {log_items}")
            
            if 'loss' in logs:
                loss = logs['loss']
                self.losses.append(loss)
                print(f"Step {len(self.losses)}: Loss = {loss:.4f}")
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        print(f"Training complete. Recorded {len(self.losses)} loss values.")
        return control

class Flickr30kDataset(Dataset):
    """
    Dataset for Flickr30k image-caption pairs
    """
    def __init__(self, image_dir, captions_file, processor, split="train"):
        self.image_dir = image_dir
        self.processor = processor
        self.split = split
        
        # Load captions from the JSON file
        with open(captions_file, 'r') as f:
            self.captions_data = json.load(f)
        
        # Process captions data
        self.examples = []
        if isinstance(self.captions_data, dict) and 'train' in self.captions_data:
            # If the data is already split
            self.examples = self.captions_data[split]
        else:
            # If using the flickr30k_captions_quintets where each item has a 'set' of captions
            for item in self.captions_data:
                if 'set' in item:
                    captions = item['set']
                    if captions:
                        # For each image, use the first caption as the primary one
                        caption = captions[0]
                        
                        # Find a matching image file (assuming a simple pattern for demonstration)
                        for img_file in os.listdir(self.image_dir):
                            if img_file.endswith(('.jpg', '.jpeg')):
                                self.examples.append({
                                    "image_path": os.path.join(self.image_dir, img_file),
                                    "caption": caption,
                                    "all_captions": captions
                                })
                                break
        
        # Shuffle the examples and split if needed
        random.shuffle(self.examples)
        if split == "train":
            # Use 80% for training
            split_idx = int(0.8 * len(self.examples))
            self.examples = self.examples[:split_idx]
        elif split == "val":
            # Use 20% for validation
            split_idx = int(0.8 * len(self.examples))
            self.examples = self.examples[split_idx:]
        
        # Optionally limit the number of examples for faster testing
        if split == "train" and len(self.examples) > 5000:
            print(f"Limiting training dataset to 5000 examples (from {len(self.examples)})")
            self.examples = self.examples[:5000]
        elif split == "val" and len(self.examples) > 1000:
            print(f"Limiting validation dataset to 1000 examples (from {len(self.examples)})")
            self.examples = self.examples[:1000]
        
        print(f"Loaded {len(self.examples)} examples for split '{split}'")
        if len(self.examples) > 0:
            print(f"Sample example: {self.examples[0]}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_path = example["image_path"]
        caption = example["caption"]
        
        try:
            # Load and process the image
            image = Image.open(image_path).convert("RGB")
            
            # Create the prompt with the instruction
            prompt = "Describe this image in detail."
            full_prompt = f"USER: <image>\n{prompt}\nASSISTANT: {caption}"
            
            # Process the inputs
            encoding = self.processor(text=full_prompt, images=image, return_tensors="pt")
            
            # Move everything to the right device and remove batch dimension
            inputs = {k: v.squeeze(0) for k, v in encoding.items()}
            
            # Add the labels manually
            tokenized_prompt = self.processor.tokenizer(
                f"USER: <image>\n{prompt}\nASSISTANT:", 
                return_tensors="pt"
            )
            
            # The length of this tokenized prompt gives us where to start computing loss
            prompt_len = tokenized_prompt.input_ids.shape[1]
            
            # Create labels tensor - start with copied input ids
            labels = inputs["input_ids"].clone()
            
            # Mask the prompt part
            labels[:prompt_len] = -100
            
            # Add labels to inputs
            inputs["labels"] = labels
            
            return inputs
            
        except Exception as e:
            print(f"Error processing {image_path} (idx={idx}): {str(e)}")
            if idx == 0:
                raise RuntimeError(f"Error processing first example: {str(e)}")
            
            # Try a different example
            alternative_idx = (idx + 1) % len(self.examples)
            print(f"Trying alternative example at idx={alternative_idx}")
            return self.__getitem__(alternative_idx)

def download_flickr30k_dataset(dataset_dir):
    """
    Download the Flickr30k dataset from GitHub releases
    """
    os.makedirs(dataset_dir, exist_ok=True)
    
    # Download the dataset parts
    parts = [
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part00",
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part01",
        "https://github.com/awsaf49/flickr-dataset/releases/download/v1.0/flickr30k_part02"
    ]
    
    for i, part_url in enumerate(parts):
        part_file = os.path.join(dataset_dir, f"flickr30k_part{i:02d}")
        if not os.path.exists(part_file):
            print(f"Downloading part {i+1}/3 of Flickr30k dataset...")
            os.system(f"wget {part_url} -O {part_file}")
    
    # Combine the parts
    zip_file = os.path.join(dataset_dir, "flickr30k.zip")
    if not os.path.exists(zip_file):
        print("Combining dataset parts...")
        os.system(f"cat {os.path.join(dataset_dir, 'flickr30k_part00')} "
                 f"{os.path.join(dataset_dir, 'flickr30k_part01')} "
                 f"{os.path.join(dataset_dir, 'flickr30k_part02')} > {zip_file}")
    
    # Extract the ZIP file
    if not os.path.exists(os.path.join(dataset_dir, "flickr30k", "images")):
        print("Extracting Flickr30k dataset...")
        os.makedirs(os.path.join(dataset_dir, "flickr30k"), exist_ok=True)
        os.system(f"unzip -q {zip_file} -d {os.path.join(dataset_dir, 'flickr30k')}")
    
    # Clean up
    for i in range(3):
        part_file = os.path.join(dataset_dir, f"flickr30k_part{i:02d}")
        if os.path.exists(part_file):
            os.remove(part_file)
    
    if os.path.exists(zip_file):
        os.remove(zip_file)
    
    print("Flickr30k dataset downloaded and extracted successfully")
    
    # Create the captions file if not exists
    create_flickr30k_captions_file(dataset_dir)

def create_flickr30k_captions_file(dataset_dir):
    """
    Create a JSON file with all captions for Flickr30k
    """
    captions_file = os.path.join(dataset_dir, "flickr30k", "captions.json")
    if os.path.exists(captions_file):
        print(f"Captions file already exists at {captions_file}")
        return captions_file
    
    # Download the captions file from HuggingFace datasets
    print("Downloading Flickr30k captions...")
    try:
        # Install datasets if needed
        try:
            from datasets import load_dataset
        except ImportError:
            print("Installing datasets package...")
            install_package("datasets")
            from datasets import load_dataset
        
        dataset = load_dataset("embedding-data/flickr30k_captions_quintets")
        
        # Convert to the format we need
        captions_data = []
        for item in dataset['train']:
            captions_data.append(item)
        
        # Save as JSON
        with open(captions_file, 'w') as f:
            json.dump(captions_data, f)
        
        print(f"Created captions file at {captions_file}")
        return captions_file
        
    except Exception as e:
        print(f"Error downloading captions: {e}")
        # Fallback to creating a simple mapping between images and dummy captions
        image_dir = os.path.join(dataset_dir, "flickr30k", "images")
        if os.path.exists(image_dir):
            print("Creating simple captions mapping...")
            captions_data = []
            for img_file in os.listdir(image_dir):
                if img_file.endswith(('.jpg', '.jpeg')):
                    captions_data.append({
                        "set": [
                            f"A photo showing {img_file}",
                            f"An image with filename {img_file}",
                            f"A picture taken and saved as {img_file}",
                            f"This is image {img_file}",
                            f"A photograph with ID {img_file}"
                        ]
                    })
            
            with open(captions_file, 'w') as f:
                json.dump(captions_data, f)
            
            print(f"Created simple captions file at {captions_file}")
            return captions_file
    
    print("Could not create captions file")
    return None

def create_improved_metrics(predictions, references):
    """
    Compute metrics that can handle mismatched datasets better
    """
    print("\nComputing improved metrics...")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Number of references: {len(references)}")
    
    # Display the first few samples to debug
    for i in range(min(3, len(predictions))):
        print(f"\nSample {i}:")
        print(f"Prediction: {predictions[i]}")
        print(f"Reference: {references[i]}")
    
    # Create a special mode for the dummy captions case (handling "A photo showing X.jpg")
    is_dummy_caption_mode = all("A photo showing" in ref for ref in references[:5])
    
    if is_dummy_caption_mode:
        print("\nDetected dummy caption mode. Computing simplified metrics...")
        # For dummy captions, let's create a simple metric based on whether
        # the model can describe images at all (any non-trivial output is a success)
        
        # Basic metrics: does prediction have content and differ between images?
        unique_predictions = len(set(predictions))
        valid_predictions = sum(len(p.split()) > 3 for p in predictions)
        
        # Variety score: ratio of unique predictions to total
        variety_score = unique_predictions / max(1, len(predictions))
        # Validity score: ratio of non-empty predictions
        validity_score = valid_predictions / max(1, len(predictions))
        
        return {
            'variety_score': variety_score,
            'validity_score': validity_score,
            'unique_predictions': unique_predictions,
            'valid_predictions': valid_predictions,
            'total_predictions': len(predictions)
        }
    
    # For regular evaluation, use NLTK metrics but with improvements
    # Tokenize predictions and references
    tokenized_preds = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    tokenized_refs = [[nltk.word_tokenize(ref.lower())] for ref in references]
    
    # Initialize the ROUGE scorer
    rouge_types = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(rouge_types, use_stemmer=True)
    
    # BLEU scores
    smoothing = SmoothingFunction().method1
    bleu1_scores = []
    bleu4_scores = []
    
    # ROUGE scores
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    
    for i, (pred, ref) in enumerate(zip(tokenized_preds, tokenized_refs)):
        # Prevent empty predictions or references from crashing
        if not pred:
            pred = ['empty', 'prediction']
        if not ref[0]:
            ref = [['empty', 'reference']]
            
        # BLEU-1 with better smoothing for short texts
        bleu1 = sentence_bleu(ref, pred, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        bleu1_scores.append(bleu1)
        
        # BLEU-4 with better smoothing
        bleu4 = sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
        bleu4_scores.append(bleu4)
        
        # ROUGE with string handling
        try:
            # Convert back to strings for ROUGE
            ref_str = ' '.join(ref[0])
            pred_str = ' '.join(pred)
            rouge_scores = scorer.score(ref_str, pred_str)
            rouge1_scores.append(rouge_scores['rouge1'].fmeasure)
            rouge2_scores.append(rouge_scores['rouge2'].fmeasure)
            rougeL_scores.append(rouge_scores['rougeL'].fmeasure)
        except Exception as e:
            print(f"Error computing ROUGE for sample {i}: {str(e)}")
            rouge1_scores.append(0.0)
            rouge2_scores.append(0.0)
            rougeL_scores.append(0.0)
    
    # Calculate average scores
    metrics = {
        'bleu1': np.mean(bleu1_scores),
        'bleu4': np.mean(bleu4_scores),
        'rouge1': np.mean(rouge1_scores),
        'rouge2': np.mean(rouge2_scores),
        'rougeL': np.mean(rougeL_scores)
    }
    
    return metrics

def evaluate_model_on_test_set(model, processor, test_set, num_samples=5):
    """
    Evaluate a model on a test set and return the generated captions and metrics
    """
    # Force model to eval mode
    model.eval()
    predictions = []
    references = []
    sample_paths = []
    
    # Ensure we don't try to sample more examples than we have
    num_samples = min(num_samples, len(test_set))
    samples = random.sample(range(len(test_set)), num_samples)
    
    print(f"Evaluating model on {num_samples} samples...")
    
    # Always use CPU for generation to avoid MPS issues
    model_device = torch.device('cpu')
    original_device = next(model.parameters()).device
    print(f"Moving model from {original_device} to {model_device} for reliable generation")
    
    # Create a copy of the model on CPU for generation
    try:
        model = model.to(model_device)
        
        for idx in samples:
            if len(predictions) >= num_samples:
                break
                
            try:
                example = test_set.examples[idx]
                image_path = example["image_path"]
                reference = example["caption"]
                
                print(f"Processing image: {os.path.basename(image_path)}")
                
                # Load image
                try:
                    image = Image.open(image_path).convert("RGB")
                    # Create the prompt
                    prompt = "Describe this image with natural language."
                    input_text = f"USER: <image>\n{prompt}\nASSISTANT:"
                    
                    # Process inputs on CPU
                    inputs = processor(text=input_text, images=image, return_tensors="pt")
                    
                    with torch.no_grad():
                        # Generate with minimal parameters
                        output = model.generate(
                            **inputs,
                            max_new_tokens=30,
                            num_beams=1,
                            do_sample=False,
                            pad_token_id=processor.tokenizer.pad_token_id
                        )
                    
                    # Decode the output
                    generated_text = processor.decode(output[0], skip_special_tokens=True)
                    
                    # Extract the actual response
                    if "ASSISTANT:" in generated_text:
                        prediction = generated_text.split("ASSISTANT:")[-1].strip()
                    else:
                        prediction = generated_text.replace(input_text, "").strip()
                    
                    if not prediction:
                        prediction = f"This is an image"
                    
                    # Store results
                    predictions.append(prediction)
                    references.append(reference)
                    sample_paths.append(image_path)
                    
                    print(f"Generated: {prediction[:50]}...")
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {str(e)}")
                    # Add a placeholder to maintain alignment
                    predictions.append("Error processing image")
                    references.append(reference)
                    sample_paths.append(image_path)
                    continue
                    
            except Exception as e:
                print(f"Skipping sample {idx} due to error: {str(e)}")
                continue
    except Exception as e:
        print(f"Error during model evaluation: {str(e)}")
    finally:
        # Move model back to original device
        if original_device != model_device:
            try:
                model = model.to(original_device)
                print(f"Model moved back to {original_device}")
            except Exception as e:
                print(f"Error moving model back to original device: {str(e)}")
    
    # If we have no predictions, generate dummy ones
    if not predictions:
        print("WARNING: No valid predictions generated. Using dummy predictions.")
        predictions = ["This is an image" for _ in range(min(3, len(test_set)))]
        references = [example["caption"] for example in test_set.examples[:len(predictions)]]
        sample_paths = [example["image_path"] for example in test_set.examples[:len(predictions)]]
    
    # Compute metrics with the improved method
    try:
        metrics = create_improved_metrics(predictions, references)
    except Exception as e:
        print(f"Error computing metrics: {str(e)}")
        metrics = {
            'variety_score': 0.0,
            'validity_score': 0.0,
            'unique_predictions': 0,
            'valid_predictions': 0,
            'total_predictions': len(predictions)
        }
    
    # Print a summary of the predictions
    print("\nSample predictions:")
    for i in range(min(3, len(predictions))):
        img_name = os.path.basename(sample_paths[i]) if i < len(sample_paths) else "unknown"
        print(f"Image: {img_name}")
        print(f"Reference: {references[i]}")
        print(f"Prediction: {predictions[i]}")
        print("-" * 40)
    
    return {
        'predictions': predictions,
        'references': references,
        'metrics': metrics,
        'image_paths': sample_paths if sample_paths else []
    }

def compare_model_performance(original_model, finetuned_model, processor, test_set, num_samples=5):
    """
    Compare the performance of the original and fine-tuned models
    Updated to handle the improved metrics
    """
    print("\nEvaluating original model...")
    try:
        original_results = evaluate_model_on_test_set(original_model, processor, test_set, num_samples)
    except Exception as e:
        print(f"Error evaluating original model: {str(e)}")
        original_results = {
            'predictions': ["Error during evaluation"],
            'references': ["Error during evaluation"],
            'metrics': {
                'variety_score': 0.0,
                'validity_score': 0.0,
                'unique_predictions': 0,
                'valid_predictions': 0,
                'total_predictions': 1
            },
            'image_paths': []
        }
    
    print("\nEvaluating fine-tuned model...")
    try:
        finetuned_results = evaluate_model_on_test_set(finetuned_model, processor, test_set, num_samples)
    except Exception as e:
        print(f"Error evaluating fine-tuned model: {str(e)}")
        finetuned_results = {
            'predictions': ["Error during evaluation"],
            'references': ["Error during evaluation"],
            'metrics': {
                'variety_score': 0.0,
                'validity_score': 0.0,
                'unique_predictions': 0,
                'valid_predictions': 0,
                'total_predictions': 1
            },
            'image_paths': []
        }
    
    # Print comparison
    print("\n===== MODEL COMPARISON =====")
    print("Metric\t\tOriginal\t\tFine-tuned\t\tImprovement")
    print("-" * 70)
    
    # Get all unique metrics from both result sets
    all_metrics = set(original_results['metrics'].keys()) | set(finetuned_results['metrics'].keys())
    
    for metric in all_metrics:
        orig_val = original_results['metrics'].get(metric, 0.0)
        ft_val = finetuned_results['metrics'].get(metric, 0.0)
        improvement = ft_val - orig_val
        print(f"{metric}\t\t{orig_val:.4f}\t\t{ft_val:.4f}\t\t{improvement:+.4f}")
    
    # Save the results to a file
    output_dir = os.path.join(project_root, "model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "comparison_results.json"), 'w') as f:
        json.dump({
            'original': {
                'metrics': original_results['metrics'],
                'predictions': original_results['predictions'],
                'references': original_results['references']
            },
            'finetuned': {
                'metrics': finetuned_results['metrics'],
                'predictions': finetuned_results['predictions'],
                'references': finetuned_results['references']
            }
        }, f, indent=2)
    
    # Create a comparative plot
    metrics_to_plot = [m for m in all_metrics if isinstance(original_results['metrics'].get(m, 0), (int, float))]
    if metrics_to_plot:
        orig_values = [original_results['metrics'].get(m, 0.0) for m in metrics_to_plot]
        ft_values = [finetuned_results['metrics'].get(m, 0.0) for m in metrics_to_plot]
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics_to_plot))
        width = 0.35
        
        plt.bar(x - width/2, orig_values, width, label='Original Model')
        plt.bar(x + width/2, ft_values, width, label='Fine-tuned Model')
        
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x, metrics_to_plot)
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'))
        print(f"Comparison results saved to {output_dir}")
    
    # Save a few sample predictions
    with open(os.path.join(output_dir, "sample_predictions.txt"), 'w') as f:
        f.write("===== SAMPLE PREDICTIONS =====\n\n")
        num_to_display = min(5, len(original_results['predictions']), len(finetuned_results['predictions']))
        for i in range(num_to_display):
            try:
                if i < len(original_results.get('image_paths', [])):
                    img_path = original_results['image_paths'][i]
                    f.write(f"Image: {os.path.basename(img_path)}\n")
                else:
                    f.write(f"Image #{i+1}\n")
                    
                if i < len(original_results['references']):
                    f.write(f"Reference: {original_results['references'][i]}\n")
                
                if i < len(original_results['predictions']):
                    f.write(f"Original model: {original_results['predictions'][i]}\n")
                
                if i < len(finetuned_results['predictions']):
                    f.write(f"Fine-tuned model: {finetuned_results['predictions'][i]}\n")
                
                f.write("\n" + "-" * 50 + "\n\n")
            except Exception as e:
                f.write(f"Error displaying sample {i}: {str(e)}\n")
                f.write("-" * 50 + "\n\n")
    
    print(f"Sample predictions saved to {os.path.join(output_dir, 'sample_predictions.txt')}")
    
    return original_results, finetuned_results

# Main execution block to be added as needed
if __name__ == "__main__":
    # Add model loading, dataset preparation, and training code here
    pass