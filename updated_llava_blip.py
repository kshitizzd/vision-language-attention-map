import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    LlavaProcessor,
    LlavaForConditionalGeneration
)

class VisionAttentionVisualizer:
    def __init__(self, model_type="blip", model_name=None):
        """
        Initialize the vision attention visualizer with specified model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model_type = model_type.lower()
        
        if self.model_type == "blip":
            if model_name is None:
                model_name = "Salesforce/blip-vqa-base"
            self.processor = BlipProcessor.from_pretrained(model_name)
            self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(self.device)
        
        elif self.model_type == "llava":
            if model_name is None:
                model_name = "llava-hf/llava-1.5-7b-hf"
            self.processor = LlavaProcessor.from_pretrained(model_name)
            self.model = LlavaForConditionalGeneration.from_pretrained(model_name).to(self.device)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Choose 'blip' or 'llava'.")
        
        self.model.eval()
        self.activations = None
        self.gradients = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register hooks to capture activations and gradients."""
        
        def forward_hook(module, input, output):
            if isinstance(output, (tuple, list)):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, (tuple, list)):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        if self.model_type == "blip":
            target_layer = self.model.vision_model.embeddings
        elif self.model_type == "llava":
            target_layer = self.model.vision_tower.vision_model.embeddings
        
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)
    
    def _process_gradcam(self, width, height):
        if self.activations is None or self.gradients is None:
            raise ValueError("Activations or gradients not captured.")
        
        act_shape = self.activations.shape
        
        if len(act_shape) == 3:
            start_idx = 1 if act_shape[1] > 196 else 0
            grad_weights = torch.mean(self.gradients[:, start_idx:, :], dim=2)
            weighted_acts = torch.mul(grad_weights.unsqueeze(-1), self.activations[:, start_idx:, :])
            cam = torch.sum(weighted_acts, dim=2)[0]
            grid_size = int(np.sqrt(cam.shape[0]))
            cam = cam.reshape(grid_size, grid_size)
        
        elif len(act_shape) == 4:
            weights = torch.mean(self.gradients, dim=[2, 3])[0]
            cam = torch.zeros(act_shape[2:]).to(self.device)
            for i, w in enumerate(weights):
                cam += w * self.activations[0, i, :, :]
        
        else:
            raise ValueError(f"Unexpected activation shape: {act_shape}")
        
        cam = torch.nn.functional.relu(cam)
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (width, height))
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        
        return cam
    
    def visualize_attention(self, image_path, query="What is in this image?", alpha=0.6, save_path=None):
        """
        Process an image and generate attention visualization.
        """
        self.activations = None
        self.gradients = None
        
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        with torch.set_grad_enabled(True):
            if self.model_type == "blip":
                inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
                
                outputs = self.model(
                    input_ids=inputs.input_ids,
                    pixel_values=inputs.pixel_values,
                    decoder_input_ids=inputs.input_ids,
                    return_dict=True
                )
                
                if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                    target = outputs.decoder_hidden_states[-1].mean()
                elif hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
                    target = outputs.text_embeds.mean()
                elif hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                    target = outputs.image_embeds.mean()
                else:
                    raise ValueError("No suitable output for backpropagation.")
            
            elif self.model_type == "llava":
                # REALLY FINAL formatting: Insert <image> token!
                formatted_query = f"USER: <image> {query}\nASSISTANT:"
                inputs = self.processor(images=image, text=formatted_query, return_tensors="pt").to(self.device)

                outputs = self.model(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    pixel_values=inputs.pixel_values,
                    return_dict=True
                )

                if hasattr(outputs, 'logits'):
                    target = outputs.logits[0, 0].sum()
                else:
                    target = outputs.last_hidden_state.mean()
            
            target.backward()
        
        cam = self._process_gradcam(width=original_size[0], height=original_size[1])
        
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        original_img = np.array(image)
        
        if heatmap.shape[:2] != original_img.shape[:2]:
            heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        
        overlaid_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(original_img)
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title(f"{self.model_type.upper()} Attention Heatmap")
        plt.imshow(heatmap)
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title(f"Overlaid Attention\nQuery: '{query}'")
        plt.imshow(overlaid_img)
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        
        return original_img, heatmap, overlaid_img

    def get_model_answer(self, image_path, query="What is in this image?"):
        """
        Get the model's answer to a query about an image.
        """
        image = Image.open(image_path).convert("RGB")
        
        if self.model_type == "blip":
            inputs = self.processor(images=image, text=query, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        elif self.model_type == "llava":
            # Insert <image> token for LLaVA input too
            formatted_query = f"USER: <image> {query}\nASSISTANT:"
            inputs = self.processor(images=image, text=formatted_query, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs)
            answer = self.processor.decode(outputs[0], skip_special_tokens=True)
            if answer.startswith(query):
                answer = answer[len(query):].strip()
        
        return answer

# --- Example Usage ---

if __name__ == "__main__":
    image_path = "/Users/matiwosbirbo/vision-language-attention-map/download.jpeg"  # Replace this path
    
    print("Initializing BLIP model...")
    blip_visualizer = VisionAttentionVisualizer(model_type="blip")
    
    print("Generating BLIP attention visualization...")
    blip_visualizer.visualize_attention(
        image_path=image_path,
        query="What is in this image?",
        save_path="blip_attention.png"
    )
    
    blip_answer = blip_visualizer.get_model_answer(
        image_path=image_path,
        query="What is in this image?"
    )
    print(f"BLIP answer: {blip_answer}")
    
    print("\nInitializing LLaVA model...")
    llava_visualizer = VisionAttentionVisualizer(model_type="llava")
    
    print("Generating LLaVA attention visualization...")
    llava_visualizer.visualize_attention(
        image_path=image_path,
        query="What is in this image?",
        save_path="llava_attention.png"
    )
    
    llava_answer = llava_visualizer.get_model_answer(
        image_path=image_path,
        query="What is in this image?"
    )
    print(f"LLaVA answer: {llava_answer}")
    
    print("\nGenerating comparison visualization...")
    plt.figure(figsize=(10, 5))
    
    _, _, blip_overlaid = blip_visualizer.visualize_attention(
        image_path=image_path,
        query="What is in this image?"
    )
    
    _, _, llava_overlaid = llava_visualizer.visualize_attention(
        image_path=image_path,
        query="What is in this image?"
    )
    
    plt.subplot(1, 2, 1)
    plt.title(f"BLIP Attention\nAnswer: '{blip_answer}'")
    plt.imshow(blip_overlaid)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"LLaVA Attention\nAnswer: '{llava_answer}'")
    plt.imshow(llava_overlaid)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.show()
