import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import traceback
from transformers import (
    BlipProcessor,
    BlipForQuestionAnswering,
    LlavaProcessor,
    LlavaForConditionalGeneration
)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import os

class VisionAttentionVisualizer:
    """
    A class for visualizing attention maps of vision-language models (BLIP and LLaVA)
    for a given image and question. It supports generating the model's answer
    and overlaying the attention heatmap on the original image.
    """
    def __init__(self, model_type="llava", model_name=None):
        """
        Initialize the vision attention visualizer with the specified model type and name.

        Args:
            model_type (str, optional): The type of model to use ('blip' or 'llava').
                Defaults to "llava".
            model_name (str, optional): The name or path of the pretrained model.
                If None, default model names are used.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model_type = model_type.lower()
        
        # Determine data type for model based on device (float16 for GPU, float32 for CPU)
        use_low_mem = True
        if self.device.type == "cuda":
            dtype = torch.float16
            print(f"Using float16 for model on GPU")
            # Print available GPU memory
            try:
                print(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
                print(f"Current allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            except Exception as e:
                print(f"Could not get GPU memory info: {e}")
        else:
            dtype = torch.float32
            print(f"Using float32 for model on CPU")
        
        try:
            if self.model_type == "blip":
                if model_name is None:
                    model_name = "Salesforce/blip-vqa-base"
                print(f"Loading BLIP processor from {model_name}")
                self.processor = BlipProcessor.from_pretrained(model_name)
                
                # Setup model loading options for memory efficiency 
                model_kwargs = {
                    'torch_dtype': dtype,
                }
                
                if use_low_mem:
                    model_kwargs['low_cpu_mem_usage'] = True
                    
                    if self.device.type == "cuda":
                        try:
                            model_kwargs['device_map'] = 'auto'
                            print(f"Using device_map='auto' for efficient memory usage")
                        except Exception as e:
                            print(f"Cannot use device_map: {e}")
                
                print(f"Loading BLIP model with options: {model_kwargs}")
                self.model = BlipForQuestionAnswering.from_pretrained(model_name, **model_kwargs)
                
                # Only manually move to device if not using device_map
                if 'device_map' not in model_kwargs:
                    self.model.to(self.device)
                    print(f"Model moved to {self.device}")
                
            elif self.model_type == "llava":
                if model_name is None:
                    model_name = "llava-hf/llava-1.5-7b-hf"
                print(f"Loading LLaVA processor from {model_name}")
                self.processor = LlavaProcessor.from_pretrained(model_name)
                
                # Setup model loading options for memory efficiency
                model_kwargs = {
                    'torch_dtype': dtype,
                }
                
                if use_low_mem:
                    model_kwargs['low_cpu_mem_usage'] = True
                    
                    if self.device.type == "cuda":
                        try:
                            model_kwargs['device_map'] = 'auto'
                            print(f"Using device_map='auto' for efficient memory usage")
                            
                            # Try to enable 8-bit loading if bitsandbytes is available
                            try:
                                import bitsandbytes as bnb
                                print("bitsandbytes available, trying 8-bit loading")
                                model_kwargs['load_in_8bit'] = True
                            except ImportError:
                                print("bitsandbytes not installed, using full precision")
                        except Exception as e:
                            print(f"Cannot use device_map: {e}")
                
                print(f"Loading LLaVA model with options: {model_kwargs}")
                self.model = LlavaForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
                
                # Only manually move to device if not using device_map
                if 'device_map' not in model_kwargs:
                    self.model.to(self.device)
                    print(f"Model moved to {self.device}")
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}. Choose 'blip' or 'llava'.")
            
            self.model.eval()
            print(f"Model set to evaluation mode")
            self.activations = None
            self.gradients = None
            
            self._register_hooks()
            print(f"{self.model_type.upper()} model initialized successfully")
            
            # Print memory usage after initialization if on CUDA for monitoring
            if self.device.type == "cuda":
                print(f"GPU memory allocated after init: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"GPU memory reserved after init: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
                
        except Exception as e:
            print(f"ERROR initializing {self.model_type} model: {e}")
            traceback.print_exc()
            raise
    
    def _register_hooks(self):
        """Register forward and backward hooks to capture activations and gradients
        from a specific layer in the vision model.
        """
        print(f"Registering hooks for {self.model_type} model")
        
        def forward_hook(module, input, output):
            """Forward hook to store the output (activations) of the target layer."""
            if isinstance(output, (tuple, list)):
                self.activations = output[0].detach()
            else:
                self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            """Backward hook to store the gradients of the output of the target layer."""
            if isinstance(grad_output, (tuple, list)):
                self.gradients = grad_output[0].detach()
            else:
                self.gradients = grad_output.detach()
        
        try:
            if self.model_type == "blip":
                # Target the embedding layer of the BLIP vision model
                target_layer = self.model.vision_model.embeddings
                print("Registered hooks on BLIP vision model embeddings")
            elif self.model_type == "llava":
                # Target the embedding layer of the LLaVA vision tower
                target_layer = self.model.vision_tower.vision_model.embeddings
                print("Registered hooks on LLaVA vision tower embeddings")
            
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
        except Exception as e:
            print(f"ERROR registering hooks: {e}")
            traceback.print_exc()
    
    def _process_gradcam(self, width, height):
        """
        Process the captured activations and gradients to generate a Grad-CAM
        style attention map.

        Args:
            width (int): The width to resize the attention map to.
            height (int): The height to resize the attention map to.

        Returns:
            numpy.ndarray: The processed attention map.

        Raises:
            ValueError: If activations or gradients have not been captured.
        """
        if self.activations is None or self.gradients is None:
            raise ValueError("Activations or gradients not captured.")
        
        act_shape = self.activations.shape
        print(f"Processing Grad-CAM with activation shape: {act_shape}")
        
        try:
            if len(act_shape) == 3:
                # For models like BLIP where the output is [batch, tokens, features]
                # We might need to skip the CLS token
                start_idx = 1 if act_shape[1] > 196 else 0 # Heuristic for excluding CLS token
                grad_weights = torch.mean(self.gradients[:, start_idx:, :], dim=2)
                weighted_acts = torch.mul(grad_weights.unsqueeze(-1), self.activations[:, start_idx:, :])
                cam = torch.sum(weighted_acts, dim=2)[0]
                grid_size = int(np.sqrt(cam.shape[0]))
                cam = cam.reshape(grid_size, grid_size)
            
            elif len(act_shape) == 4:
                # For models where the output is [batch, channels, height, width]
                weights = torch.mean(self.gradients, dim=[2, 3])[0]
                cam = torch.zeros(act_shape[2:]).to(self.device)
                for i, w in enumerate(weights):
                    cam += w * self.activations[0, i, :, :]
            
            else:
                raise ValueError(f"Unexpected activation shape: {act_shape}")

            # Apply ReLU to only consider positive influences
            cam = torch.nn.functional.relu(cam)
            cam = cam.cpu().numpy()
            # Resize the attention map to the original image dimensions
            cam = cv2.resize(cam, (width, height))
            # Normalize the attention map to the range [0, 1]
            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
            
            return cam
        except Exception as e:
            print(f"ERROR in Grad-CAM processing: {e}")
            traceback.print_exc()
            # Return a fallback attention map
            print("WARNING: Using fallback random attention map")
            return np.random.rand(width, height)
    
    def process_image_and_question(self, image_path, question):
        """
        Process the input image and question to get the model's answer and
        generate an attention map highlighting relevant regions in the image
        for the answer.

        Args:
            image_path (str): Path to the input image.
            question (str): The question related to the image.

        Returns:
            tuple: (original PIL Image, model's answer (str), attention map (numpy.ndarray))

        Main processing logic:
            1. Load image
            2. Get model answer
            3. Register hooks and run forward/backward for attention
            4. Generate attention map
            5. Return results

        Print statements for error/debug only below this point
        """
        print(f"\nProcessing image {image_path} with question: '{question}'")
        try:
            image = Image.open(image_path).convert("RGB")
            print(f"Image loaded: {image.size} - Mode: {image.mode}")
            answer = self.get_model_answer(image_path, question)
            print(f"Generated answer: {answer}")
            
            # Perform attention visualization to get the attention map
            self.activations = None
            self.gradients = None

            # Enable gradient calculation for backpropagation
            with torch.set_grad_enabled(True):
                if self.model_type == "blip":
                    print("Processing with BLIP for attention visualization")
                    inputs = self.processor(images=image, text=question, return_tensors="pt")
                    # Move inputs to correct device
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(self.device)

                    # Perform a forward pass
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        pixel_values=inputs.pixel_values,
                        decoder_input_ids=inputs.input_ids,
                        return_dict=True
                    )

                    # Determine the target tensor for backpropagation. This might vary
                    # based on the model's output structure.
                    if hasattr(outputs, 'decoder_hidden_states') and outputs.decoder_hidden_states is not None:
                        target = outputs.decoder_hidden_states[-1].mean()
                    elif hasattr(outputs, 'text_embeds') and outputs.text_embeds is not None:
                        target = outputs.text_embeds.mean()
                    elif hasattr(outputs, 'image_embeds') and outputs.image_embeds is not None:
                        target = outputs.image_embeds.mean()
                    else:
                        raise ValueError("No suitable output for backpropagation.")
                
                elif self.model_type == "llava":
                    print("Processing with LLaVA for attention visualization")
                    formatted_query = f"USER: <image> {question}\nASSISTANT:"
                    inputs = self.processor(images=image, text=formatted_query, return_tensors="pt")
                    # Move inputs to correct device
                    for key in inputs:
                        if isinstance(inputs[key], torch.Tensor):
                            inputs[key] = inputs[key].to(self.device)

                    # Perform a forward pass
                    outputs = self.model(
                        input_ids=inputs.input_ids,
                        attention_mask=inputs.attention_mask,
                        pixel_values=inputs.pixel_values,
                        return_dict=True
                    )

                    # Determine the target tensor for backpropagation.
                    if hasattr(outputs, 'logits'):
                        target = outputs.logits[0, 0].sum() # Sum of the first token's logits
                    else:
                        target = outputs.last_hidden_state.mean() # Fallback to mean of last hidden state
                
                print("Calculating gradients for attention visualization")
                target.backward() # Perform backpropagation to calculate gradients
            
            # Generate the attention map using the captured activations and gradients
            original_size = image.size
            print(f"Generating attention map for image size: {original_size}")
            attention_map = self._process_gradcam(width=original_size[0], height=original_size[1])
            print("Attention map generation complete")
            
            return image, answer, attention_map
        
        except Exception as e:
            print(f"ERROR in process_image_and_question: {e}")
            traceback.print_exc()
            # Return a mock result to avoid crashing
            fallback_img = Image.new('RGB', (300, 300), color='gray')
            fallback_map = np.random.rand(300, 300)
            return fallback_img, f"Error: {str(e)}", fallback_map

    def visualize_attention(self, image, attention_map, alpha=0.6):
        """
        Overlay the generated attention heatmap on the original image.

        Args:
            image (PIL.Image.Image): The original image.
            attention_map (numpy.ndarray): The attention map to overlay.
            alpha (float, optional): The transparency of the heatmap (0 to 1).
                Defaults to 0.6.

        Returns:
            PIL.Image.Image: The image with the attention heatmap overlaid.
        """
        try:
            original_img = np.array(image)

            # Apply a colormap to the attention map to create a heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * attention_map), cv2.COLORMAP_JET)
            # Convert the heatmap to RGB format
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            # Resize the heatmap to match the original image dimensions if necessary
            if heatmap.shape[:2] != original_img.shape[:2]:
                print(f"Resizing heatmap from {heatmap.shape[:2]} to {original_img.shape[:2]}")
                heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))

             # Overlay the heatmap on the original image with specified transparency
            overlaid_img = cv2.addWeighted(original_img, 1 - alpha, heatmap, alpha, 0)
            
            return Image.fromarray(overlaid_img)
        except Exception as e:
            print(f"ERROR in visualize_attention: {e}")
            traceback.print_exc()
            # Return the original image if visualization fails
            return image

    def get_model_answer(self, image_path, query="What is in this image?"):
        """
        Get the model's answer to a given question about the image.

        Args:
            image_path (str): Path to the input image.
            query (str, optional): The question to ask about the image.
                Defaults to "What is in this image?".

        Returns:
            str: The model's generated answer to the question.
        """
        try:
            print(f"Getting model answer for query: '{query}'")
            image = Image.open(image_path).convert("RGB")
            print(f"Image loaded successfully: {image.size}")
            
            # Check if image is too large and resize if needed
            max_size = 1024
            if max(image.size) > max_size:
                print(f"Resizing large image from {image.size}", end="")
                ratio = min(max_size / image.width, max_size / image.height)
                new_size = (int(image.width * ratio), int(image.height * ratio))
                image = image.resize(new_size)
                print(f" to {image.size}")
            
            if self.model_type == "blip":
                print("Processing with BLIP")
                inputs = self.processor(images=image, text=query, return_tensors="pt")
                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                print("Generating answer...")
                with torch.no_grad():
                    outputs = self.model.generate(**inputs)
                    
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                print(f"BLIP generated answer: {answer}")
            
            elif self.model_type == "llava":
                print("Processing with LLaVA")
                # Insert <image> token for LLaVA input
                formatted_query = f"USER: <image> {query}\nASSISTANT:"
                inputs = self.processor(images=image, text=formatted_query, return_tensors="pt")
                # Move inputs to device
                for key in inputs:
                    if isinstance(inputs[key], torch.Tensor):
                        inputs[key] = inputs[key].to(self.device)
                
                print("Generating answer...")
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=200)
                
                answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                # Remove the original query from the start of the answer if present
                if answer.startswith(query):
                    answer = answer[len(query):].strip()
                print(f"LLaVA generated answer: {answer}")
            
            return answer
        except Exception as e:
            print(f"ERROR in get_model_answer: {e}")
            traceback.print_exc()
            return f"Error generating answer: {str(e)}"
    
    def evaluate_answer(self, generated_answer, reference_answers):
        if isinstance(reference_answers, str):
            reference_answers = [reference_answers]

        reference_tokens = [ref.split() for ref in reference_answers]
        candidate_tokens = generated_answer.split()

        smoothie = SmoothingFunction().method4
        bleu = sentence_bleu(reference_tokens, candidate_tokens, smoothing_function=smoothie)

        scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
        rouge_scores = [scorer.score(ref, generated_answer) for ref in reference_answers]

        avg_rouge1 = np.mean([score['rouge1'].fmeasure for score in rouge_scores])
        avg_rougeL = np.mean([score['rougeL'].fmeasure for score in rouge_scores])

        return {
            "bleu": bleu,
            "rouge1": avg_rouge1,
            "rougeL": avg_rougeL
        }

if __name__ == "__main__":
    # Example usage
    import os
    
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two directories to get to the project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    # Path to the test image
    image_path = os.path.join(project_root, "flower.jpg")

    # Initialize the visualizer with the LLaVA model
    visualizer = VisionAttentionVisualizer(model_type="llava")
    # Define the question to ask about the image
    question = "What is the main object in this image?"
    reference = "A sigle pink rose in a garden setting."
    
    print(f"Processing image: {image_path}")
    # Process the image and question to get the original image, the model's answer, and the attention map
    image, answer, attention_map = visualizer.process_image_and_question(image_path, question)
    # Overlay the attention map on the original image to create the visualization
    attention_visualization = visualizer.visualize_attention(image, attention_map)
    
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    
    # Define the output directory for saving the visualization
    output_dir = os.path.join(project_root, "outputs")
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Display the original image and the attention visualization
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
    plt.savefig(os.path.join(output_dir, "vision_attention_visualization.png"))
    print(f"Visualization saved to {os.path.join(output_dir, 'vision_attention_visualization.png')}")
    plt.show() 
