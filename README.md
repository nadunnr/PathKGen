# Vision-Language Model for Histopathology Report Generation

Complete implementation of a VLM that generates pathology reports from WSI slide features using Qwen 2.5 with SlideChat weights.

## Architecture

```
WSI → TITAN Encoder → [768-dim features]
                            ↓
                      Projection Layer (MLP)
                            ↓
                      [32 visual tokens × 3584-dim]
                            ↓
                      Qwen 2.5 7B (SlideChat weights)
                            ↓
                      Generated Report
```

## Features

- ✅ **Projection Layer**: Maps 768-dim TITAN features to 32 visual tokens in LLM space
- ✅ **SlideChat Integration**: Loads pathology-specific weights from SlideChat
- ✅ **Two-Stage Training**:
  - Stage 1: Train projection only (LLM frozen)
  - Stage 2: Joint training with QLoRA (memory-efficient)
- ✅ **QLoRA Support**: 4-bit quantization for RTX 3090 (24GB VRAM)
- ✅ **Efficient Data Pipeline**: Custom collator with proper padding
- ✅ **Inference Tools**: Single slide and batch generation

## Installation

```bash
# Install dependencies
pip install torch transformers peft bitsandbytes accelerate
pip install h5py tqdm matplotlib

# Verify installation
python pipeline/model.py  # Should run test successfully
```

## Quick Start

### 1. Test Model Loading

```bash
# Test the model architecture
python pipeline/model.py
```

Expected output:
```
Testing VLM Model
[Stage 1 Test] Projection layer training
Total parameters: 7,615,XXX,XXX
Trainable parameters: ~2,XXX,XXX
✓ All tests passed!
```

### 2. Stage 1 Training (Projection Only)

```bash
python pipeline/train_stage1.py \
    --data_json /home/nadun/wd/vlm_histo/data/curated_breast_dataset.json \
    --epochs 10 \
    --batch_size 8 \
    --lr 1e-4 \
    --output_dir /home/nadun/wd/vlm_histo/checkpoints/stage1
```

**Memory**: ~12-14GB VRAM  
**Time**: ~3-5 hours on RTX 3090  
**Batch size**: 8-16 (adjust based on sequence length)

### 3. Stage 2 Training (Projection + QLoRA)

```bash
python pipeline/train_stage2.py \
    --stage1_checkpoint /home/nadun/wd/vlm_histo/checkpoints/stage1/best_model.pt \
    --data_json /home/nadun/wd/vlm_histo/data/curated_breast_dataset.json \
    --epochs 5 \
    --batch_size 2 \
    --lr 2e-5 \
    --projection_lr 1e-4 \
    --gradient_accumulation_steps 4 \
    --output_dir /home/nadun/wd/vlm_histo/checkpoints/stage2
```

**Memory**: ~18-20GB VRAM  
**Time**: ~6-12 hours on RTX 3090  
**Batch size**: 2-4 with gradient accumulation

### 4. Inference

**Single Slide:**
```bash
python pipeline/inference.py \
    --checkpoint /home/nadun/wd/vlm_histo/checkpoints/stage2/best_model.pt \
    --feature_path /path/to/slide_features.h5 \
    --output_path report.txt \
    --use_qlora  # If using Stage 2 checkpoint
```

**Batch Inference:**
```bash
python pipeline/inference.py \
    --checkpoint /home/nadun/wd/vlm_histo/checkpoints/stage2/best_model.pt \
    --data_json /home/nadun/wd/vlm_histo/data/curated_breast_dataset.json \
    --split test \
    --output_dir /home/nadun/wd/vlm_histo/results/test_reports \
    --save_json \
    --use_qlora
```

### 5. Visualize Training

```bash
python pipeline/plot_training.py \
    --stage1_history /home/nadun/wd/vlm_histo/checkpoints/stage1/training_history.json \
    --stage2_history /home/nadun/wd/vlm_histo/checkpoints/stage2/training_history.json \
    --output_dir /home/nadun/wd/vlm_histo/plots
```

## File Structure

```
pipeline/
├── model.py              # VLM architecture (ProjectionLayer + VisionLanguageModel)
├── dataset.py            # Dataset loader for WSI features and reports
├── data_collator.py      # Custom collator for batching
├── train_stage1.py       # Stage 1 training script
├── train_stage2.py       # Stage 2 training script (QLoRA)
├── inference.py          # Inference script
├── plot_training.py      # Training visualization
└── language_model.py     # (legacy) SlideChat loading example
```

## Configuration

### Model Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `visual_dim` | 768 | TITAN encoder output dimension |
| `num_visual_tokens` | 32 | Number of visual tokens for LLM |
| `llm_hidden_dim` | 3584 | Qwen 2.5 7B hidden dimension |

### Training Hyperparameters

**Stage 1:**
- Learning rate: 1e-4
- Batch size: 8-16
- Epochs: 10
- Optimizer: AdamW
- Scheduler: CosineAnnealing

**Stage 2:**
- LLM learning rate: 2e-5
- Projection learning rate: 1e-4
- Batch size: 2-4
- Gradient accumulation: 4
- Epochs: 5
- LoRA rank: 64, alpha: 128

### QLoRA Configuration

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                   "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
)
```

## Dataset Format

Your dataset should follow the format in [`dataset.py`](file:///home/nadun/wd/vlm_histo/pipeline/dataset.py):

**JSON structure:**
```json
{
  "train": [
    {
      "id": "slide_001",
      "organ": "breast",
      "feature_path": "/path/to/slide_001_features.h5",
      "report": "Pathology report text..."
    }
  ],
  "val": [...],
  "test": [...]
}
```

**H5 feature files:**
- Key: `"features"` (configurable via `--feature_key`)
- Shape: `[768]` (TITAN encoder output)
- Dtype: `float32`

## Future Work

- [ ] Integrate Knowledge Graph tokens (green pathway in architecture diagram)
- [ ] Multi-organ support with organ-specific prompts
- [ ] Evaluation metrics (BLEU, ROUGE, medical entity F1)
- [ ] Interactive demo interface
- [ ] Model quantization for deployment

## Troubleshooting

### OOM (Out of Memory) Errors

**Stage 1:**
- Reduce batch size: `--batch_size 4`
- Enable gradient accumulation: `--gradient_accumulation_steps 2`

**Stage 2:**
- Already uses 4-bit quantization
- Reduce batch size: `--batch_size 1`
- Increase gradient accumulation: `--gradient_accumulation_steps 8`
- Reduce `num_visual_tokens` to 16

### SlideChat Weights Not Loading

Check that the checkpoint path is correct:
```bash
ls /home/nadun/wd/vlm_histo/checkpoints/SlideChat_Weight/stage2_pth/mp_rank_00_model_states.pt
```

If missing, the model will fall back to base Qwen 2.5 weights.

### Slow Training

- Use larger batch size if memory allows
- Reduce `num_workers` if CPU bottleneck
- Use gradient accumulation instead of increasing batch size
- Enable mixed precision training (already enabled with `bfloat16`)

## References

- [SlideChat](https://github.com/uni-medical/SlideChat) - Pathology VLM
- [Qwen 2.5](https://github.com/QwenLM/Qwen2.5) - Base LLM
- [QLoRA](https://github.com/artidoro/qlora) - Efficient fine-tuning
- [PEFT](https://github.com/huggingface/peft) - Parameter-Efficient Fine-Tuning

## License

This project builds upon SlideChat and Qwen 2.5. Please refer to their respective licenses.

## Citation

If you use this code, please cite:

```bibtex
@article{slidechat2024,
  title={SlideChat: A Large Vision-Language Assistant for Whole-Slide Pathology Image Understanding},
  author={...},
  journal={CVPR},
  year={2025}
}
```
