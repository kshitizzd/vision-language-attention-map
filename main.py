#!/usr/bin/env python
"""
Vision-Language Attention Map Visualizer

This is the main entry point for running the application.
It provides options to run just the vision attention analysis, 
text attention analysis, or the complete application.
"""

import argparse
import sys
import os
from PIL import Image
from transformers import pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Vision-Language Attention Map Visualizer"
    )
    
    parser.add_argument(
        "--mode", 
        type=str,
        choices=["vision", "text", "app", "all", "pipeline", "blip"],
        default="app",
        help="Which component to run: vision (LLaVA), text, app, pipeline, blip, or all"
    )
    
    parser.add_argument(
        "--image", 
        type=str,
        default="flower.jpg",
        help="Path to image file for vision or text analysis"
    )
    
    parser.add_argument(
        "--question", 
        type=str,
        default="What is the main object in this image?",
        help="Question about the image for vision or text analysis"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable detailed debugging output"
    )
    
    parser.add_argument(
        "--force-cpu", 
        action="store_true",
        help="Force using CPU even if CUDA is available"
    )
    
    args = parser.parse_args()
    
    # Get project root directory (where main.py is located)
    project_root = os.path.dirname(os.path.abspath(__file__))
    print(f"Project root: {project_root}")
    
    # Force CPU if requested
    if args.force_cpu:
        print("Forcing CPU mode as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    # Resolve image path (if it's a relative path)
    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    
    # Create temp directory if it doesn't exist
    os.makedirs(os.path.join(project_root, "temp"), exist_ok=True)
    
    if args.debug:
        print("\nENVIRONMENT INFO:")
        print(f"Python version: {sys.version}")
        
        try:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                print(f"CUDA version: {torch.version.cuda}")
                print(f"GPU: {torch.cuda.get_device_name(0)}")
                print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        except:
            print("PyTorch not available")
            
        try:
            import transformers
            print(f"Transformers version: {transformers.__version__}")
        except:
            print("Transformers not available")
            
        print(f"\nImage path exists: {os.path.exists(image_path)}")
        print(f"Working directory: {os.getcwd()}")
        print("\n")
    
    if args.mode == "pipeline" or args.mode == "all":
        print("Running LLaVA Pipeline Demo...")
        
        try:
            # Load the image
            print(f"Loading image from path: {image_path}")
            image = Image.open(image_path)
            print(f"Image loaded: {image.size} - Mode: {image.mode}")
            
            # Resize image if it's too large (LLaVA has size limits)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                print(f"Resizing image from {image.size}", end="")
                ratio = min(max_size / image.width, max_size / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size)
                print(f" to {image.size}")
            
            # Save resized image for debugging
            debug_path = os.path.join(project_root, "temp", "debug_resized.jpg")
            image.save(debug_path)
            print(f"Saved debug image to {debug_path}")
            
            # Direct model approach instead of pipeline
            print("Loading LLaVA model and processor directly...")
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            import torch
            
            # Set device
            if args.force_cpu:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Load model and processor with low memory usage options
            model_id = "llava-hf/llava-1.5-7b-hf"
            print(f"Loading model: {model_id}")
            
            # Add memory-saving options
            use_low_mem = True
            dtype = torch.float16 if device == "cuda" else torch.float32
            
            processor = LlavaProcessor.from_pretrained(model_id)
            print("Processor loaded successfully")
            
            model_kwargs = {
                'torch_dtype': dtype,
                'low_cpu_mem_usage': use_low_mem,
            }
            
            if use_low_mem and device == "cuda":
                try:
                    print("Attempting to load with device_map='auto' for memory efficiency")
                    model_kwargs['device_map'] = 'auto'
                    # Try to enable 8-bit loading if available
                    try:
                        import bitsandbytes as bnb
                        print("bitsandbytes available, trying 8-bit loading")
                        model_kwargs['load_in_8bit'] = True
                    except ImportError:
                        print("bitsandbytes not installed, using full precision")
                except Exception as e:
                    print(f"Warning: Could not use device_map: {e}")
            
            print(f"Loading model with parameters: {model_kwargs}")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_id,
                **model_kwargs
            )
            print("Model loaded successfully")
            
            # Only move to device explicitly if not using device_map='auto'
            if 'device_map' not in model_kwargs:
                model.to(device)
                print(f"Model moved to {device}")
            
            # Format the prompt according to LLaVA requirements
            formatted_query = f"USER: <image> {args.question}\nASSISTANT:"
            print(f"Formatted query: {formatted_query}")
            
            # Process inputs
            print("Processing inputs...")
            inputs = processor(text=formatted_query, images=image, return_tensors="pt")
            
            # Move inputs to the correct device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)
            
            # Print debug info about inputs
            for key, value in inputs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
            
            # Check memory usage if on CUDA
            if device == "cuda":
                print(f"Memory allocated before generation: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"Memory reserved before generation: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # Generate
            print("Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
            
            # Decode output
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            print("\nResults:")
            print(f"Question: {args.question}")
            print(f"Answer: {generated_text}")
            
            # Save the output
            output_file = os.path.join(project_root, "outputs", "llava_output.txt")
            with open(output_file, "w") as f:
                f.write(f"Question: {args.question}\n")
                f.write(f"Answer: {generated_text}")
            print(f"Output saved to {output_file}")
            
        except Exception as e:
            print(f"\nError in pipeline mode: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode == "vision" or args.mode == "all":
        print("Running LLaVA Vision Attention Analysis...")
        try:
            from src.vision.vision_attention import VisionAttentionVisualizer
            
            print("Initializing LLaVA visualizer...")
            visualizer = VisionAttentionVisualizer(model_type="llava")
            print(f"Loading image from: {image_path}")
            image, answer, attention_map = visualizer.process_image_and_question(image_path, args.question)
            print(f"Image processed, generating visualization...")
            attention_visualization = visualizer.visualize_attention(image, attention_map)
            
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
            
            # Save visualization
            output_file = os.path.join(project_root, "outputs", "llava_attention.jpg")
            attention_visualization.save(output_file)
            print(f"Vision attention visualization saved to {output_file}")
        except Exception as e:
            print(f"\nError in vision mode: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode == "blip" or args.mode == "all":
        print("Running BLIP Vision Attention Analysis...")
        try:
            from src.vision.vision_attention import VisionAttentionVisualizer
            
            print("Initializing BLIP visualizer...")
            visualizer = VisionAttentionVisualizer(model_type="blip")
            print(f"Loading image from: {image_path}")
            image, answer, attention_map = visualizer.process_image_and_question(image_path, args.question)
            print(f"Image processed, generating visualization...")
            attention_visualization = visualizer.visualize_attention(image, attention_map)
            
            print(f"Question: {args.question}")
            print(f"Answer: {answer}")
            
            # Save visualization
            output_file = os.path.join(project_root, "outputs", "blip_attention.jpg")
            attention_visualization.save(output_file)
            print(f"BLIP attention visualization saved to {output_file}")
        except Exception as e:
            print(f"\nError in BLIP mode: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode == "text" or args.mode == "all":
        print("Running Text Attention Analysis...")
        try:
            from src.text.text_attention import TextAttentionVisualizer
            import matplotlib.pyplot as plt
            
            visualizer = TextAttentionVisualizer()
            print(f"Processing image and question: {image_path}, {args.question}")
            answer, tokenized_question, token_attentions = visualizer.process_image_and_question(
                image_path, args.question
            )
            
            print(f"Question: {args.question}")
            print(f"Tokenized: {tokenized_question}")
            print(f"Answer: {answer}")
            
            # Visualize token attention
            print("Generating token attention visualization...")
            fig = visualizer.visualize_token_attention(tokenized_question, token_attentions)
            output_file = os.path.join(project_root, "outputs", "text_attention.jpg")
            fig.savefig(output_file)
            plt.close(fig)
            print(f"Text attention visualization saved to {output_file}")
        except Exception as e:
            print(f"\nError in text mode: {e}")
            import traceback
            traceback.print_exc()
    
    if args.mode == "app" or args.mode == "all":
        print("Starting the full application...")
        try:
            # Save current working directory
            original_cwd = os.getcwd()
            
            # First modify environment for app if needed
            if args.force_cpu:
                # Force CPU for the UI app too
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
                print("Forcing CPU mode for UI app")
            
            # Change directory to UI directory for proper CSS path resolution
            ui_dir = os.path.join(project_root, "src", "ui")
            os.chdir(ui_dir)
            print(f"Changed working directory to: {ui_dir}")
            
            # Import the demo with the correct path
            sys.path.append(os.path.join(project_root, "src", "ui"))
            from src.ui.app import demo
            
            # Launch the demo with the project root as an allowed path
            demo.launch(allowed_paths=[project_root], debug=True)
            
            # Restore original working directory
            os.chdir(original_cwd)
        except Exception as e:
            print(f"\nError in app mode: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 