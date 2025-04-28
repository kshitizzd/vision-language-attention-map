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

def main():
    parser = argparse.ArgumentParser(
        description="Vision-Language Attention Map Visualizer"
    )
    
    parser.add_argument(
        "--mode", 
        type=str,
        choices=["vision", "text", "app", "all"],
        default="app",
        help="Which component to run: vision, text, app, or all"
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
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    # Create temp directory if it doesn't exist
    os.makedirs("temp", exist_ok=True)
    
    if args.mode == "vision" or args.mode == "all":
        print("Running Vision Attention Analysis...")
        from src.vision.vision_attention import VisionAttentionVisualizer
        
        visualizer = VisionAttentionVisualizer()
        image, answer, attention_map = visualizer.process_image_and_question(args.image, args.question)
        attention_visualization = visualizer.visualize_attention(image, attention_map)
        
        print(f"Question: {args.question}")
        print(f"Answer: {answer}")
        
        # Save visualization
        attention_visualization.save("outputs/vision_attention.jpg")
        print("Vision attention visualization saved to outputs/vision_attention.jpg")
    
    if args.mode == "text" or args.mode == "all":
        print("Running Text Attention Analysis...")
        from src.text.text_attention import TextAttentionVisualizer
        import matplotlib.pyplot as plt
        
        visualizer = TextAttentionVisualizer()
        answer, tokenized_question, token_attentions = visualizer.process_image_and_question(
            args.image, args.question
        )
        
        print(f"Question: {args.question}")
        print(f"Tokenized: {tokenized_question}")
        print(f"Answer: {answer}")
        
        # Visualize token attention
        fig = visualizer.visualize_token_attention(tokenized_question, token_attentions)
        fig.savefig("outputs/text_attention.jpg")
        plt.close(fig)
        print("Text attention visualization saved to outputs/text_attention.jpg")
    
    if args.mode == "app" or args.mode == "all":
        print("Starting the full application...")
        # Change directory to UI directory for proper CSS path resolution
        os.chdir(os.path.join("src", "ui"))
        from src.ui.app import demo
        demo.launch()

if __name__ == "__main__":
    main() 