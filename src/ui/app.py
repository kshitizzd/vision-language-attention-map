import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import torch
import os
import time
import sys
import traceback
import gc

# Add parent directory to path to import modules from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.vision_attention import VisionAttentionVisualizer
from text.text_attention import TextAttentionVisualizer

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Get project root directory (2 levels up from current file)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
print(f"Project root: {project_root}")

# Create outputs directory if it doesn't exist
os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
os.makedirs(os.path.join(project_root, "temp"), exist_ok=True)

# Initialize visualizers
try:
    print("Initializing visualizers...")
    
    # Initialize vision visualizer
    vision_model_name = "llava-hf/llava-1.5-7b-hf"
    print(f"Creating vision visualizer with {vision_model_name}")
    vision_visualizer = VisionAttentionVisualizer(model_name=vision_model_name)
    
    # Initialize text visualizer with a BERT model since it's more suitable for text attention
    text_model_name = "google-bert/bert-base-uncased"
    print(f"Creating text visualizer with {text_model_name}")
    text_visualizer = TextAttentionVisualizer(model_name=text_model_name)
    
    print("Visualizers initialized successfully")
except Exception as e:
    print(f"ERROR initializing visualizers: {e}")
    traceback.print_exc()

def figure_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    return image

def process_image_and_question(image, question):
    """Process the image and question, extract attention visualizations."""
    print("\n--- Starting image and question processing ---")
    print(f"Image type: {type(image)}")
    print(f"Question: {question}")

    if image is None:
        print("Error: Image is None")
        return "Please upload an image first.", None, None, None, None
    
    if not question or question.strip() == "":
        print("Error: Question is empty")
        return "Please enter a question about the image.", None, None, None, None
    
    # Save the uploaded image temporarily
    temp_dir = os.path.join(project_root, "temp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_image_path = os.path.join(temp_dir, f"temp_image_{int(time.time())}.jpg")
    
    try:
        # Save image for debugging
        print(f"Saving image to {temp_image_path}")
        image.save(temp_image_path)
        print(f"Image saved successfully, size: {image.size}")
        
        print("Processing with vision attention visualizer...")
        # Process with vision attention visualizer
        original_image, answer, vision_attention_map = vision_visualizer.process_image_and_question(
            temp_image_path, question
        )
        print(f"Vision processing complete. Answer: {answer}")
        
        print("Generating vision attention image...")
        vision_attention_image = vision_visualizer.visualize_attention(
            original_image, vision_attention_map
        )
        print("Vision attention image generated")
        
        # Force garbage collection to free memory
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            print("Cleared CUDA cache after vision processing")
        
        print("Processing with text attention visualizer...")
        # Process with text attention visualizer - new implementation
        # The text_visualizer.process_image_and_question method now returns question, answer, attention_visualization
        _, text_answer, text_attention_fig = text_visualizer.process_image_and_question(
            temp_image_path, question
        )
        print("Text processing complete")
        
        if text_attention_fig:
            text_attention_image = figure_to_image(text_attention_fig)
            plt.close(text_attention_fig)  # Close the figure to free memory
            print("Text attention visualization converted to image")
        else:
            print("WARNING: No text attention figure generated, using placeholder")
            # Create a placeholder image
            placeholder_fig = plt.figure(figsize=(10, 5))
            plt.text(0.5, 0.5, "Text attention visualization not available", 
                    ha='center', va='center', fontsize=12)
            plt.axis('off')
            text_attention_image = figure_to_image(placeholder_fig)
            plt.close(placeholder_fig)
        
        print("Creating comparison visualization...")
        # Create comparison visualization
        comparison_fig = plt.figure(figsize=(14, 6))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(original_image)
        plt.title("Original Image", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Vision attention
        plt.subplot(1, 3, 2)
        plt.imshow(vision_attention_image)
        plt.title("Vision Attention", fontsize=12, fontweight='bold')
        plt.axis('off')
        
        # Add the answer as text
        plt.subplot(1, 3, 3)
        plt.text(0.5, 0.5, f"Q: {question}\n\nA: {answer}", 
                ha='center', va='center', wrap=True, fontsize=12)
        plt.axis('off')
        
        plt.tight_layout()
        comparison_image = figure_to_image(comparison_fig)
        plt.close(comparison_fig)
        print("Comparison visualization created")
        
        # Force garbage collection again
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
            print("Cleared CUDA cache after processing")
        
        print("Processing complete, returning results")
        return answer, vision_attention_image, text_attention_image, comparison_image, original_image
    
    except Exception as e:
        print(f"ERROR processing image and question: {e}")
        traceback.print_exc()
        # Return a more detailed error message
        error_msg = f"Error: {str(e)}\n\nStack Trace: {traceback.format_exc()}"
        return error_msg, None, None, None, None
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
                print(f"Temp file {temp_image_path} removed")
        except Exception as cleanup_error:
            print(f"Error removing temp file: {cleanup_error}")

# Sample questions for the examples
example_questions = [
    ["What is the main subject of this image?"],
    ["What colors do you see in this image?"],
    ["Is there any text visible in this image?"],
    ["What time of day does this image depict?"],
    ["Describe the mood or atmosphere of this scene."]
]

# Sample images for the gallery - use relative paths from current directory
# Copy the sample image to the current directory first to avoid path issues
sample_image_path = os.path.join(os.path.dirname(__file__), "sample_image.jpg")
if not os.path.exists(sample_image_path):
    # Try to copy from project root if available
    try:
        import shutil
        source_path = os.path.join(project_root, "flower.jpg")
        if os.path.exists(source_path):
            shutil.copy(source_path, sample_image_path)
            print(f"Copied sample image to {sample_image_path}")
        else:
            print(f"Warning: Sample image not found at {source_path}")
    except Exception as e:
        print(f"Warning: Could not copy sample image: {e}")

# Use the local sample image if it exists
example_images = []
if os.path.exists(sample_image_path):
    example_images = [sample_image_path]

# Create Gradio interface
with gr.Blocks(
    title="Vision-Language Attention Visualizer", 
    css="./style.css", 
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate"
    )
) as demo:
    # Header section
    with gr.Row(elem_classes="header"):
        gr.HTML("""
        <div style="text-align: center; width: 100%;">
            <h1 style="font-size: 2.8rem; font-weight: 700; margin-bottom: 0.5rem; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.2);">Vision Language Attention Map</h1>
            <h3 style="font-size: 1.3rem; margin: 0; color: rgba(255,255,255,0.9);">See how AI models perceive and understand images and text</h3>
        </div>
        """)
    
    # Main content
    with gr.Row():
        # Left panel - Input
        with gr.Column(scale=1, elem_classes="card"):
            with gr.Group():
                gr.Markdown("###  Step 1: Upload an Image")
                input_image = gr.Image(type="pil", label="", elem_classes="image-display", height=300)
                
                gr.Markdown("###  Step 2: Ask a Question")
                input_question = gr.Textbox(
                    label="", 
                    placeholder="What do you want to ask about this image?",
                    lines=2
                )
                
                with gr.Row():
                    clear_button = gr.Button("Clear", variant="secondary", elem_classes="animate-button")
                    submit_button = gr.Button(" Analyze", variant="primary", elem_classes="animate-button")
                
                gr.Markdown("### ⚡️ Try These Example Questions")
                gr.Examples(
                    example_questions,
                    inputs=input_question
                )
                
                if len(example_images) > 0:
                    gr.Markdown("###  Example Images")
                    gr.Examples(
                        example_images,
                        inputs=input_image
                    )
        
        # Right panel - Output
        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                output_answer = gr.Textbox(label="🤖 AI Model's Answer", lines=3, elem_classes="highlight")
                
                with gr.Tabs() as tabs:
                    with gr.TabItem(" Attention Overview", elem_classes="tab-item"):
                        output_comparison = gr.Image(label="", elem_classes="image-display")
                        gr.Markdown("""
                        This overview shows the original image, the attention heatmap visualization, and the model's response to your question.
                        The heatmap highlights areas the model focused on when answering your question.
                        """)
                    
                    with gr.TabItem(" Vision Attention", elem_classes="tab-item"):
                        with gr.Row():
                            output_original = gr.Image(label="Original Image", elem_classes="image-display")
                            output_vision_attention = gr.Image(label="Attention Heatmap", elem_classes="image-display")
                        
                        with gr.Accordion("What is Vision Attention?", open=False):
                            gr.Markdown("""
                            The colored overlay shows regions of the image that the model focused on most when answering your question:
                            
                            - **Red/Yellow areas**: Received the most attention from the model
                            - **Blue/Green areas**: Received less attention
                            
                            This visualization helps you understand which parts of the image the AI considered important for answering your specific question.
                            """)
                    
                    with gr.TabItem(" Text Attention", elem_classes="tab-item"):
                        output_text_attention = gr.Image(label="", elem_classes="image-display")
                        with gr.Accordion("What is Text Attention?", open=False):
                            gr.Markdown("""
                            This heatmap shows how tokens in your question relate to each other when the model processes your text:
                            
                            - **Brighter colors**: Indicate stronger attention between tokens
                            - **Darker colors**: Indicate weaker attention between tokens
                            
                            The heatmap shows which words influence each other in your question, giving insight into how the language model processes and understands your query.
                            """)
    
    # Error output for debugging
    debug_output = gr.Textbox(label="Debug Information (Error Details)", visible=False, lines=10)
    
    # Footer
    with gr.Row(elem_classes="footer"):
        gr.HTML("""
        <div style="text-align: center; width: 100%;">
            <p>This app visualizes how vision-language models (LLaVA 1.5) attend to different parts of images and text when answering questions.</p>
            <p>Vision Transformer + Language Model Attention Visualization Project</p>
            <div class="tooltip">
                About the Models
                <span class="tooltip-text">Vision attention uses LLaVA 1.5 (Large Language and Vision Assistant), while text attention uses BERT, a powerful language model for understanding text relationships.</span>
            </div>
        </div>
        """)
    
    # Debug button
    debug_button = gr.Button("Show Debug Info", visible=False)
    
    # Set up event handlers
    def process_with_debug(image, question):
        try:
            answer, vision_img, text_img, comp_img, orig_img = process_image_and_question(image, question)
            if isinstance(answer, str) and answer.startswith("Error:"):
                # Show the debug output if there's an error
                return answer, vision_img, text_img, comp_img, orig_img, answer, gr.update(visible=True)
            return answer, vision_img, text_img, comp_img, orig_img, "", gr.update(visible=False)
        except Exception as e:
            error_msg = f"Unhandled error: {str(e)}\n\n{traceback.format_exc()}"
            return f"Error occurred: {str(e)}", None, None, None, None, error_msg, gr.update(visible=True)
    
    submit_button.click(
        fn=process_with_debug,
        inputs=[input_image, input_question],
        outputs=[output_answer, output_vision_attention, output_text_attention, output_comparison, output_original, debug_output, debug_output],
    )
    
    def toggle_debug():
        return gr.update(visible=not debug_output.visible)
    
    debug_button.click(
        fn=toggle_debug,
        inputs=[],
        outputs=[debug_output],
    )
    
    clear_button.click(
        lambda: (None, "", None, None, None, None, "", gr.update(visible=False)),
        inputs=[],
        outputs=[input_image, input_question, output_answer, output_vision_attention, output_text_attention, output_comparison, output_original, debug_output],
    )

if __name__ == "__main__":
    # Launch with allowed paths to fix file access issues
    debug_mode = True
    demo.launch(allowed_paths=[project_root], debug=debug_mode) 
