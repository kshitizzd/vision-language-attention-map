# Vision-Language Attention Map Visualizer

This project visualizes how vision-language models attend to different parts of images and text when answering questions about images.

## Project Structure

The project is organized into three main modules to allow for parallel development by team members:

```
vision-language-attention-map/
├── src/
│   ├── vision/           # Vision Attention Module
│   │   └── vision_attention.py
│   ├── text/             # Text & Answer Module
│   │   └── text_attention.py
│   └── ui/               # UI & System Integration
│       ├── app.py
│       └── style.css
├── outputs/              # Saved visualizations
├── temp/                 # Temporary files
├── flower.jpg            # Sample image
├── main.py               # Main entry point
└── requirements.txt      # Dependencies
```

## Team Assignments

### 1. Vision Attention Module (`src/vision/`)

- Handles image input and visualizes how the model attends to different regions
- Extracts attention weights from vision layers and generates heatmaps
- Main file: `vision_attention.py`

### 2. Text & Answer Module (`src/text/`)

- Handles question input and generates answers
- Visualizes token-level attention in questions
- Main file: `text_attention.py`

### 3. UI & System Integration (`src/ui/`)

- Creates user interface for uploading images and asking questions
- Integrates vision and text modules into a unified system
- Main files: `app.py` and `style.css`

## Getting Started

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/vision-language-attention-map.git
   cd vision-language-attention-map
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

You can run the application in different modes:

1. Full application with UI:

   ```bash
   python main.py
   ```

2. Vision attention analysis only:

   ```bash
   python main.py --mode vision --image flower.jpg --question "What is the main object in this image?"
   ```

3. Text attention analysis only:

   ```bash
   python main.py --mode text --image flower.jpg --question "What is the main object in this image?"
   ```

4. Both analyses without UI:
   ```bash
   python main.py --mode all --image flower.jpg --question "What is the main object in this image?"
   ```

## Development Guides

### Vision Attention Module Developer

1. Focus on improving the vision attention extraction in `src/vision/vision_attention.py`
2. Enhance the heatmap visualization techniques
3. Test with different vision models and attention mechanisms

### Text Attention Module Developer

1. Focus on improving the text attention extraction in `src/text/text_attention.py`
2. Enhance token importance visualization
3. Experiment with different attention aggregation techniques

### UI & Integration Developer

1. Focus on improving the UI in `src/ui/app.py` and `src/ui/style.css`
2. Ensure smooth integration of vision and text modules
3. Add features like visualization history, comparison views, etc.

## Technical Details

- The project uses LLaVA 1.5, a vision-language model combining CLIP and language model capabilities
- For vision attention, we extract cross-attention between vision and text modalities
- For text attention, we analyze self-attention patterns in the language model component

## Notes

- The attention extraction methods are currently approximations
- For a production-ready application, these methods may need customization based on specific model architectures
- The model requires significant GPU memory (recommend 8+ GB VRAM for optimal performance)

## Features

- **Vision Attention Visualization**: See which regions of the image the model focuses on when generating answers
- **Text Token Attention**: Visualize which words in your question are most important to the model
- **Interactive UI**: Upload images and ask questions through a user-friendly interface

## Future Improvements

- Fine-tuning the model on specific datasets
- Adding more advanced attention visualization techniques
- Supporting more vision-language models

## Credits

This project uses the LLaVA 1.5 model, which combines CLIP ViT-L/336px and Vicuna for visual grounding in conversations.
