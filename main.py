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
    # Force CPU if requested
    if args.force_cpu:
        print("Forcing CPU mode as requested")
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Resolve image path (if it's a relative path)
    image_path = args.image
    if not os.path.isabs(image_path):
        image_path = os.path.join(project_root, image_path)
    # Create output and temp directories if they don't exist
    os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
    os.makedirs(os.path.join(project_root, "temp"), exist_ok=True)
    # Print debug info if requested
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
    # Pipeline mode: direct LLaVA pipeline demo
    if args.mode == "pipeline" or args.mode == "all":
        try:
            # Load the image
            image = Image.open(image_path)
            # Resize image if it's too large (LLaVA has size limits)
            max_size = 1024
            if image.width > max_size or image.height > max_size:
                ratio = min(max_size / image.width, max_size / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size)
            # Save resized image for debugging
            debug_path = os.path.join(project_root, "temp", "debug_resized.jpg")
            image.save(debug_path)
            # Direct model approach instead of pipeline
            from transformers import LlavaProcessor, LlavaForConditionalGeneration
            import torch
            # Set device
            if args.force_cpu:
                device = "cpu"
            else:
                device = "cuda" if torch.cuda.is_available() else "cpu"
            # Load model and processor with low memory usage options
            model_id = "llava-hf/llava-1.5-7b-hf"
            processor = LlavaProcessor.from_pretrained(model_id)
            model_kwargs = {
                'torch_dtype': torch.float16 if device == "cuda" else torch.float32,
                'low_cpu_mem_usage': True,
            }
            if device == "cuda":
                try:
                    model_kwargs['device_map'] = 'auto'
                    try:
                        import bitsandbytes as bnb
                        model_kwargs['load_in_8bit'] = True
                    except ImportError:
                        pass
                except Exception:
                    pass
            model = LlavaForConditionalGeneration.from_pretrained(model_id, **model_kwargs)
            if 'device_map' not in model_kwargs:
                model.to(device)
            # Format the prompt according to LLaVA requirements
            formatted_query = f"USER: <image> {args.question}\nASSISTANT:"
            # Process inputs
            inputs = processor(text=formatted_query, images=image, return_tensors="pt")
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(device)
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False
                )
            # Decode output
            generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
            # Save the output
            output_file = os.path.join(project_root, "outputs", "llava_output.txt")
            with open(output_file, "w") as f:
                f.write(f"Question: {args.question}\n")
                f.write(f"Answer: {generated_text}")
        except Exception as e:
            print(f"\nError in pipeline mode: {e}")
            import traceback
            traceback.print_exc()
    # Vision mode: LLaVA vision attention analysis
    if args.mode == "vision" or args.mode == "all":
        try:
            from src.vision.vision_attention import VisionAttentionVisualizer
            visualizer = VisionAttentionVisualizer(model_type="llava")
            image, answer, attention_map = visualizer.process_image_and_question(image_path, args.question)
            attention_visualization = visualizer.visualize_attention(image, attention_map)
            # Save visualization
            output_file = os.path.join(project_root, "outputs", "llava_attention.jpg")
            attention_visualization.save(output_file)
        except Exception as e:
            print(f"\nError in vision mode: {e}")
            import traceback
            traceback.print_exc()
    # BLIP mode: BLIP vision attention analysis
    if args.mode == "blip" or args.mode == "all":
        try:
            from src.vision.vision_attention import VisionAttentionVisualizer
            visualizer = VisionAttentionVisualizer(model_type="blip")
            image, answer, attention_map = visualizer.process_image_and_question(image_path, args.question)
            attention_visualization = visualizer.visualize_attention(image, attention_map)
            # Save visualization
            output_file = os.path.join(project_root, "outputs", "blip_attention.jpg")
            attention_visualization.save(output_file)
        except Exception as e:
            print(f"\nError in BLIP mode: {e}")
            import traceback
            traceback.print_exc()
    # Text mode: text attention analysis
    if args.mode == "text" or args.mode == "all":
        try:
            from src.text.text_attention import TextAttentionVisualizer
            import matplotlib.pyplot as plt
            visualizer = TextAttentionVisualizer()
            answer, tokenized_question, token_attentions = visualizer.process_image_and_question(
                image_path, args.question
            )
            # Visualize token attention
            fig = visualizer.visualize_token_attention(tokenized_question, token_attentions)
            output_file = os.path.join(project_root, "outputs", "text_attention.jpg")
            fig.savefig(output_file)
            plt.close(fig)
        except Exception as e:
            print(f"\nError in text mode: {e}")
            import traceback
            traceback.print_exc()
    # App mode: launch the Gradio app
    if args.mode == "app" or args.mode == "all":
        try:
            # Save current working directory
            original_cwd = os.getcwd()
            # First modify environment for app if needed
            if args.force_cpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
            # Change directory to UI directory for proper CSS path resolution
            ui_dir = os.path.join(project_root, "src", "ui")
            os.chdir(ui_dir)
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