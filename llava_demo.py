#!/usr/bin/env python
"""
LLaVA Demo Script

Demonstrates using LLaVA model with both:
1. High-level pipeline API
2. Direct model usage
"""

import argparse
from PIL import Image
import torch

def main():
    parser = argparse.ArgumentParser(description="LLaVA Demo")
    parser.add_argument(
        "--image", 
        type=str,
        default="sample_image.jpg",
        help="Path to image file"
    )
    parser.add_argument(
        "--question", 
        type=str,
        default="What can you see in this image?",
        help="Question about the image"
    )
    args = parser.parse_args()
    
    # Load image
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    print("="*50)
    print("DEMO 1: Using high-level pipeline API")
    print("="*50)
    
    try:
        # Use a pipeline as a high-level helper
        from transformers import pipeline
        
        # Load image first
        image = Image.open(args.image).convert("RGB")
        
        # Use the pipeline
        pipe = pipeline("image-to-text", model="llava-hf/llava-1.5-7b-hf")
        
        # Create a prompt
        prompt = f"USER: {args.question}\nASSISTANT:"
        
        # Process the image and question
        outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
        
        print(f"Question: {args.question}")
        print(f"Answer: {outputs[0]['generated_text']}")
    except Exception as e:
        print(f"Error with pipeline approach: {e}")
    
    print("\n")
    print("="*50)
    print("DEMO 2: Using model directly")
    print("="*50)
    
    try:
        # Load model directly
        from transformers import AutoProcessor, AutoModelForImageTextToText
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = AutoModelForImageTextToText.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        model.to(device)
        
        # Load the image
        raw_image = Image.open(args.image).convert("RGB")
        
        # Create conversation in the format expected by the processor
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.question},
                    {"type": "image"},
                ],
            }
        ]
        
        # Apply chat template to get correctly formatted prompt
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Process input
        inputs = processor(images=raw_image, text=prompt, return_tensors="pt").to(device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                return_dict_in_generate=True
            )
        
        # Decode output
        generated_text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0]
        
        print(f"Question: {args.question}")
        print(f"Answer: {generated_text}")
    except Exception as e:
        print(f"Error with direct model approach: {e}")

if __name__ == "__main__":
    main() 