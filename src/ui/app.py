import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import torch
import os
import time
import sys

# Add parent directory to path to import modules from sibling directories
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.vision_attention import VisionAttentionVisualizer
from text.text_attention import TextAttentionVisualizer

# Check if GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Create outputs directory if it doesn't exist
os.makedirs("../../outputs", exist_ok=True)

# Initialize visualizers
model_name = "llava-hf/llava-1.5-7b"
vision_visualizer = VisionAttentionVisualizer(model_name=model_name)
text_visualizer = TextAttentionVisualizer(model_name=model_name)

def figure_to_image(fig):
    """Convert matplotlib figure to PIL Image."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    image = Image.open(buf)
    return image

def process_image_and_question(image, question):
    """Process the image and question, extract attention visualizations."""
    if image is None:
        return "Please upload an image first.", None, None, None, None
    
    if not question or question.strip() == "":
        return "Please enter a question about the image.", None, None, None, None
    
    # Create temp directory if it doesn't exist
    if not os.path.exists("../../temp"):
        os.makedirs("../../temp")
    
    # Save the uploaded image temporarily
    temp_image_path = f"../../temp/temp_image_{int(time.time())}.jpg"
    image.save(temp_image_path)
    
    try:
        # Process with vision attention visualizer
        original_image, answer, vision_attention_map = vision_visualizer.process_image_and_question(
            temp_image_path, question
        )
        
        vision_attention_image = vision_visualizer.visualize_attention(
            original_image, vision_attention_map
        )
        
        # Process with text attention visualizer
        _, tokenized_question, token_attentions = text_visualizer.process_image_and_question(
            temp_image_path, question
        )
        
        # Create text attention visualization
        text_attention_fig = text_visualizer.visualize_token_attention(
            tokenized_question, token_attentions, figsize=(12, 5)
        )
        text_attention_image = figure_to_image(text_attention_fig)
        plt.close(text_attention_fig)  # Close the figure to free memory
        
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
        
        return answer, vision_attention_image, text_attention_image, comparison_image, original_image
    
    except Exception as e:
        print(f"Error processing image and question: {e}")
        return f"Error: {str(e)}", None, None, None, None
    finally:
        # Clean up temp file
        try:
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)
        except:
            pass

# Sample questions for the examples
example_questions = [
    ["What is the main subject of this image?"],
    ["What colors do you see in this image?"],
    ["Is there any text visible in this image?"],
    ["What time of day does this image depict?"],
    ["Describe the mood or atmosphere of this scene."]
]

# Sample images for the gallery (placeholders - you'll need actual images)
example_images = [
    "../../flower.jpg"
]

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
                gr.Markdown("### 📷 Step 1: Upload an Image")
                input_image = gr.Image(type="pil", label="", elem_classes="image-display", height=300)
                
                gr.Markdown("### 💬 Step 2: Ask a Question")
                input_question = gr.Textbox(
                    label="", 
                    placeholder="What do you want to ask about this image?",
                    lines=2
                )
                
                with gr.Row():
                    clear_button = gr.Button("🗑️ Clear", variant="secondary", elem_classes="animate-button")
                    submit_button = gr.Button("✨ Analyze", variant="primary", elem_classes="animate-button")
                
                gr.Markdown("### ⚡️ Try These Example Questions")
                gr.Examples(
                    example_questions,
                    inputs=input_question,
                    elem_classes="examples"
                )
                
                if len(example_images) > 0:
                    gr.Markdown("### 🖼️ Example Images")
                    gr.Examples(
                        example_images,
                        inputs=input_image,
                        elem_classes="examples"
                    )
        
        # Right panel - Output
        with gr.Column(scale=2):
            with gr.Group(elem_classes="card"):
                output_answer = gr.Textbox(label="🤖 AI Model's Answer", lines=3, elem_classes="highlight")
                
                with gr.Tabs() as tabs:
                    with gr.TabItem("📊 Attention Overview", elem_classes="tab-item"):
                        output_comparison = gr.Image(label="", elem_classes="image-display")
                        gr.Markdown("""
                        This overview shows the original image, the attention heatmap visualization, and the model's response to your question.
                        The heatmap highlights areas the model focused on when answering your question.
                        """)
                    
                    with gr.TabItem("👁️ Vision Attention", elem_classes="tab-item"):
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
                    
                    with gr.TabItem("📝 Text Attention", elem_classes="tab-item"):
                        output_text_attention = gr.Image(label="", elem_classes="image-display")
                        with gr.Accordion("What is Text Attention?", open=False):
                            gr.Markdown("""
                            This chart shows which words in your question were most important to the AI when generating its answer:
                            
                            - **Taller bars**: Words that received more focus from the model
                            - **Shorter bars**: Words that received less attention
                            
                            The model assigns different importance to different words in your question. This visualization helps you understand which parts of your question influenced the answer most.
                            """)
    
    # Footer
    with gr.Row(elem_classes="footer"):
        gr.HTML("""
        <div style="text-align: center; width: 100%;">
            <p>This app visualizes how vision-language models (LLaVA 1.5) attend to different parts of images and text when answering questions.</p>
            <p>Vision Transformer + Language Model Attention Visualization Project</p>
            <div class="tooltip">
                About LLaVA
                <span class="tooltip-text">LLaVA (Large Language and Vision Assistant) is a powerful vision-language model that combines CLIP's vision capabilities with an LLM for understanding and reasoning about images.</span>
            </div>
        </div>
        """)
    
    # Set up event handlers
    submit_button.click(
        fn=process_image_and_question,
        inputs=[input_image, input_question],
        outputs=[output_answer, output_vision_attention, output_text_attention, output_comparison, output_original],
    )
    
    clear_button.click(
        lambda: (None, "", None, None, None, None),
        inputs=[],
        outputs=[input_image, input_question, output_answer, output_vision_attention, output_text_attention, output_comparison, output_original],
    )

if __name__ == "__main__":
    demo.launch() 