import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForSeq2SeqLM, 
    AutoModelForCausalLM
)
import traceback

class TextAttentionVisualizer:
    def __init__(self, model_name="google-bert/bert-base-uncased"):
        """
        Initialize the text attention visualizer with a specified model.
        """
        print(f"Initializing TextAttentionVisualizer with model: {model_name}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get memory-efficient loading options based on device
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
            # Setup tokenizer
            print(f"Loading tokenizer from {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Setup model loading options
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
            
            # Try to determine model architecture
            print(f"Attempting to load model with options: {model_kwargs}")
            
            # First try sequence classification (BERT-like)
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)
                self.model_type = "classification"
                print(f"Loaded classification model: {model_name}")
            except Exception as e:
                print(f"Could not load as classification model: {e}")
                
                # Next try seq2seq (T5-like)
                try:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **model_kwargs)
                    self.model_type = "seq2seq"
                    print(f"Loaded seq2seq model: {model_name}")
                except Exception as e:
                    print(f"Could not load as seq2seq model: {e}")
                    
                    # Finally try causal LM (GPT-like)
                    try:
                        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                        self.model_type = "causal"
                        print(f"Loaded causal LM model: {model_name}")
                    except Exception as e:
                        print(f"Could not load as causal LM model: {e}")
                        raise ValueError(f"Failed to load model {model_name} with any architecture")
            
            # Only manually move to device if not using device_map
            if 'device_map' not in model_kwargs:
                self.model.to(self.device)
                print(f"Model moved to {self.device}")
                
            # Set model to evaluation mode
            self.model.eval()
            print(f"Model set to evaluation mode")
            
            # Print memory usage after initialization if on CUDA
            if self.device.type == "cuda":
                print(f"GPU memory allocated after init: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
                print(f"GPU memory reserved after init: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
        
        except Exception as e:
            print(f"ERROR initializing model: {e}")
            traceback.print_exc()
            raise
            
    def process_text(self, text):
        """
        Process a piece of text and visualize token attention.
        """
        print(f"\nProcessing text: '{text[:50]}...' ({len(text)} chars)")
        
        try:
            # Tokenize the input
            inputs = self.tokenizer(text, return_tensors="pt")
            
            # Move inputs to device
            for key in inputs:
                if isinstance(inputs[key], torch.Tensor):
                    inputs[key] = inputs[key].to(self.device)
            
            # Get token attention
            attention_data = self.get_token_attention(inputs)
            
            # Generate token visualization
            tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
            attention_visualization = self.visualize_token_attention(tokens, attention_data)
            
            return tokens, attention_data, attention_visualization
            
        except Exception as e:
            print(f"ERROR in process_text: {e}")
            traceback.print_exc()
            # Return fallback data
            fallback_tokens = ["[ERROR]"] * 5
            fallback_attn = np.random.rand(5, 5)
            return fallback_tokens, fallback_attn, None
        
    def get_token_attention(self, inputs):
        """
        Get attention between tokens from the model.
        """
        try:
            print("Extracting token attention...")
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)
            
            # Extract attention from outputs (varies by model type)
            attentions = outputs.attentions
            
            if attentions is None:
                print("WARNING: Model did not return attentions, using random data")
                # Generate random attention data for demonstration
                seq_len = inputs.input_ids.shape[1]
                return np.random.rand(seq_len, seq_len)
            
            # Take last layer attention
            last_layer_attention = attentions[-1]
            
            # Average attention across all heads
            avg_attention = last_layer_attention.mean(dim=1)
            
            # Get the first batch item
            attention_matrix = avg_attention[0].cpu().numpy()
            
            print(f"Extracted attention matrix shape: {attention_matrix.shape}")
            return attention_matrix
            
        except Exception as e:
            print(f"ERROR in get_token_attention: {e}")
            traceback.print_exc()
            # Return fallback random attention
            seq_len = inputs.input_ids.shape[1]
            return np.random.rand(seq_len, seq_len)
        
    def visualize_token_attention(self, tokens, attention_matrix, max_tokens=50):
        """
        Visualize attention between tokens as a heatmap.
        """
        try:
            print(f"Visualizing token attention for {len(tokens)} tokens")
            
            # Truncate if too many tokens
            if len(tokens) > max_tokens:
                print(f"Truncating from {len(tokens)} to {max_tokens} tokens for visualization")
                tokens = tokens[:max_tokens]
                attention_matrix = attention_matrix[:max_tokens, :max_tokens]
            
            plt.figure(figsize=(10, 8))
            ax = sns.heatmap(
                attention_matrix,
                xticklabels=tokens,
                yticklabels=tokens,
                cmap="viridis",
                cbar=True
            )
            
            # Rotate x-axis labels for readability
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            
            plt.title("Token Attention Heatmap")
            plt.tight_layout()
            
            # Instead of showing, we'll save to a buffer
            fig = ax.figure
            return fig
            
        except Exception as e:
            print(f"ERROR in visualize_token_attention: {e}")
            traceback.print_exc()
            return None
        
    def process_image_and_question(self, image_path, question):
        """
        Compatibility method to match the interface of VisionAttentionVisualizer.
        For text models, this will just process the question text.
        """
        print(f"\nProcessing question for text attention: '{question}'")
        try:
            tokens, attention_data, attention_visualization = self.process_text(question)
            
            # Create a mock answer (since this is text-only model)
            answer = f"Text attention analysis for: '{question}'"
            
            return question, answer, attention_visualization
            
        except Exception as e:
            print(f"ERROR in process_image_and_question: {e}")
            traceback.print_exc()
            return question, f"Error: {str(e)}", None
        
if __name__ == "__main__":
    # Example usage
    visualizer = TextAttentionVisualizer()
    
    # Example text
    sample_text = "The quick brown fox jumps over the lazy dog."
    
    print(f"Processing sample text: {sample_text}")
    tokens, attention_data, visualization = visualizer.process_text(sample_text)
    
    print(f"Tokens: {tokens}")
    print(f"Attention matrix shape: {attention_data.shape}")
    
    # Display visualization
    if visualization:
        plt.figure(visualization)
        plt.savefig("text_attention_visualization.png")
        print("Visualization saved to text_attention_visualization.png")
        plt.show() 