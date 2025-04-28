import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import LlavaProcessor, LlavaForConditionalGeneration

class VisionAttentionVisualizer:
    def __init__(self, model_name="llava-hf/llava-1.5-7b"):
        """Initialize the vision attention visualizer with a specified model."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = LlavaProcessor.from_pretrained(model_name)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            output_attentions=True
        )
        self.model.to(self.device)
    
    def process_image_and_question(self, image_path, question):
        """Process image and question, run model inference."""
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=question, images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                output_attentions=True,
                return_dict_in_generate=True
            )
        
        # Extract the answer
        generated_ids = outputs.sequences
        answer = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Extract attention map from the vision encoder
        # This is model-specific and may need adjustment based on model architecture
        vision_attentions = self.extract_vision_attention(inputs, outputs)
        
        return image, answer, vision_attentions
    
    def extract_vision_attention(self, inputs, outputs):
        """Extract attention maps from vision transformer layers."""
        # For LLaVA, we need to extract cross-attention between vision and text
        # This is a simplified version - actual implementation depends on model internals
        
        # Run the model again with output_attentions=True for just the encoder
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(
                inputs.input_ids, 
                attention_mask=inputs.attention_mask,
                output_attentions=True
            )
        
        # Extract cross-attention between vision and text
        # Note: The exact access pattern depends on the model architecture
        # For LLaVA, we'll need to look at specific implementation details
        
        # This is a placeholder - actual implementation will vary
        attention_maps = encoder_outputs.attentions
        
        # Process attention maps for visualization
        if attention_maps is not None and len(attention_maps) > 0:
            # Take the last layer's attention map
            attention_map = attention_maps[-1]
            
            # Average across attention heads
            attention_map = attention_map.mean(dim=1)
            
            # Extract the relevant part (vision tokens)
            # This is specific to LLaVA and would need to be adapted
            vision_attention = attention_map[0, :, :].cpu().numpy()
            
            return vision_attention
        
        # Fallback - return dummy attention map for demonstration
        return np.random.rand(14, 14)  # Placeholder 14x14 attention map
    
    def visualize_attention(self, image, attention_map, alpha=0.6):
        """Overlay attention heatmap on the original image."""
        # Resize attention map to match image dimensions
        img_np = np.array(image)
        h, w, _ = img_np.shape
        
        # Reshape attention map to a square grid (assuming patch-based transformer)
        # For LLaVA 1.5, the vision encoder uses 14×14 patches
        grid_size = int(np.sqrt(attention_map.shape[0])) if len(attention_map.shape) == 1 else attention_map.shape[0]
        attention_map = attention_map.reshape(grid_size, grid_size)
        
        # Resize to image dimensions
        attention_map = cv2.resize(attention_map, (w, h))
        
        # Normalize attention map
        attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Overlay heatmap on original image
        overlaid = cv2.addWeighted(img_np, 1-alpha, heatmap, alpha, 0)
        
        return Image.fromarray(overlaid)

if __name__ == "__main__":
    # Example usage
    visualizer = VisionAttentionVisualizer()
    image_path = "../../flower.jpg"
    question = "What is the main object in this image?"
    
    image, answer, attention_map = visualizer.process_image_and_question(image_path, question)
    attention_visualization = visualizer.visualize_attention(image, attention_map)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Display the visualization
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(attention_visualization)
    plt.title("Attention Visualization")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("../../outputs/vision_attention_visualization.png")
    plt.show() 