# LLaVA Fine-tuning Guide

This guide explains how to fine-tune the LLaVA (Large Language and Vision Assistant) model on custom data, monitor the training progress, and test the fine-tuned model.

## Initial Setup

Make sure you have activated your Python virtual environment:

```bash
source venv/bin/activate
```

## Fine-tuning the Model

The fine-tuning script is configured to use the COCO dataset for training. You can run it with:

```bash
python finetune_llava.py
```

For a quicker test using a small subset of the dataset:

```bash
python finetune_llava.py --small
```

The `--small` option will:
1. Create a smaller dataset with 100 images from COCO
2. Train for 1 epoch only
3. Save the fine-tuned model to `finetuned_llava_small/`

## Monitoring Training Progress

### Real-time Monitoring with TensorBoard

You can monitor the training progress in real-time using TensorBoard:

```bash
python monitor_training.py --output_dir finetuned_llava_small
```

This will:
1. Install TensorBoard if not already installed
2. Start TensorBoard with the correct log directory
3. Open your browser to show the live training metrics

### Creating Loss Plots After Training

After training has completed, you can generate plots of the training loss:

```bash
python plot_training_loss.py --output_dir finetuned_llava_small
```

This will:
1. Find and parse the training logs
2. Create a plot of the training loss over time
3. Save the plot to `finetuned_llava_small/training_loss_plot.png`

## Testing the Fine-tuned Model

After training completes, you can test the model on some images:

```bash
python test_finetuned_model.py --model_path finetuned_llava_small
```

Options:
- `--model_path`: Path to the fine-tuned model (default: `finetuned_llava_small`)
- `--test_dir`: Optional directory containing test images
- `--num_images`: Number of images to test (default: 5)
- `--output_dir`: Directory to save test results (default: `test_results`)

The test script will:
1. Load the fine-tuned model
2. Select random test images (from COCO val set or elsewhere)
3. Generate captions for each image
4. Create visualizations of the results
5. Save everything to the output directory

## Training Configuration

The fine-tuning process uses:
- LoRA (Low-Rank Adaptation) for efficient fine-tuning
- Mixed precision training when a GPU is available
- Small batch size (1) with gradient accumulation (4 steps)
- Learning rate of 1e-5
- One epoch training for the small dataset

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Missing dependencies**: Make sure you've installed all requirements
3. **No logs found**: Check that the log directory is correct
4. **Invalid dataset path**: Ensure your COCO dataset is in the expected location

### Memory Management

The LLaVA model is quite large. If you encounter memory issues:
- Use a smaller batch size
- Increase gradient accumulation steps
- Use CPU-only mode if GPU memory is insufficient
- Consider reducing the number of training examples

## Further Improvements

For better results, you may want to:
1. Train for more epochs
2. Use a larger and more diverse dataset
3. Tune the learning rate and other hyperparameters
4. Use a different prompt format for your specific use case 