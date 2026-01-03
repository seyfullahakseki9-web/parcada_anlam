# 🇹🇷 Turkish Reading Comprehension Fine-tuning with Qwen3-14B

Fine-tuning **Qwen3-14B** model for Turkish reading comprehension (Parçada Anlam - Passage Understanding) tasks using LoRA (Low-Rank Adaptation) on A100 GPU.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.36+-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Format](#dataset-format)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Pre-training Evaluation](#2-pre-training-evaluation)
  - [3. Fine-tuning](#3-fine-tuning)
  - [4. Post-training Evaluation](#4-post-training-evaluation)
- [Model Configuration](#model-configuration)
- [Training Configuration](#training-configuration)
- [Results](#results)
- [Hardware Requirements](#hardware-requirements)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## 🎯 Overview

This project fine-tunes the **Qwen3-14B** model for Turkish reading comprehension tasks. The model learns to:
- Read and understand Turkish passages
- Answer multiple-choice questions (A-E options)
- Provide logical explanations for answers

**Key Technique:** LoRA (Low-Rank Adaptation) for efficient fine-tuning with 4-bit quantization.

## ✨ Features

- 🚀 **Efficient Training**: LoRA + 4-bit quantization reduces trainable parameters by ~99%
- 📊 **Comprehensive Evaluation**: Detailed metrics including confusion matrix, per-class F1 scores
- 🔍 **Pre/Post Comparison**: Baseline evaluation before fine-tuning
- 💾 **Automatic Checkpointing**: Best model selection based on validation loss
- 📈 **TensorBoard Logging**: Real-time training visualization
- ⚡ **A100 Optimized**: Configuration tuned for 80GB A100 GPU
- 🎓 **Educational**: Detailed logging and analysis for learning purposes

## 📁 Dataset Format

### Training Data (`train.json`)

```json
{
  "id": "question_001",
  "metin": "Türkçe paragraf metni buraya gelir...",
  "soru": "Bu parçadan aşağıdakilerden hangisi çıkarılabilir?",
  "secenekler": [
    "A) İlk seçenek",
    "B) İkinci seçenek",
    "C) Üçüncü seçenek",
    "D) Dördüncü seçenek",
    "E) Beşinci seçenek"
  ],
  "dogru_cevap": "D",
  "aciklama": "Doğru cevabın açıklaması..."
}
```

### Test Data (`test.json`)

Same format as training data, but `aciklama` field is optional.

### Required Fields

| Field | Type | Description | Required |
|-------|------|-------------|----------|
| `metin` | string | Passage text | ✅ Yes |
| `soru` | string | Question | ✅ Yes |
| `secenekler` | list | 5 options (A-E) | ✅ Yes |
| `dogru_cevap` | string | Correct answer (A/B/C/D/E) | ✅ Yes |
| `aciklama` | string | Explanation | ⚠️ Training only |
| `id` | string | Unique identifier | ✅ Recommended |

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU support)
- 80GB+ GPU memory (A100 recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/turkish-qa-finetuning.git
cd turkish-qa-finetuning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements (`requirements.txt`)

```
torch>=2.0.0
transformers>=4.36.0
datasets>=2.14.0
peft>=0.7.0
bitsandbytes>=0.41.0
accelerate>=0.25.0
tensorboard>=2.15.0
tqdm>=4.66.0
```

## 📂 Project Structure

```
turkish-qa-finetuning/
│
├── data/
│   ├── train.json              # Training data
│   └── test.json               # Test data
│
├── scripts/
│   ├── 01_data_preparation.py  # Data loading and preprocessing
│   ├── 02_baseline_eval.py     # Pre-training evaluation
│   ├── 03_training.py          # Fine-tuning script
│   ├── 04_post_eval.py         # Post-training evaluation
│   └── 05_single_test.py       # Single example testing
│
├── models/
│   └── qwen3_parcada_lora_final/  # Saved model
│
├── logs/
│   └── tensorboard/            # TensorBoard logs
│
├── results/
│   ├── baseline_results.json   # Pre-training results
│   └── final_results.json      # Post-training results
│
├── requirements.txt
├── README.md
└── LICENSE
```

## 🚀 Usage

### 1. Data Preparation

Prepare your training and test data in JSON format:

```bash
python scripts/01_data_preparation.py
```

This script:
- Loads data from `train.json` and `test.json`
- Validates data format
- Creates tokenized datasets
- Filters invalid examples

**Output:**
```
Ham Train: 5000
Ham Test : 1000
Temiz Train: 4987
Temiz Test : 998
Train silinen: 13
Test silinen : 2
```

### 2. Pre-training Evaluation

Evaluate the base model before fine-tuning:

```bash
python scripts/02_baseline_eval.py
```

**Features:**
- Progress bar with real-time accuracy
- Checkpoint logging every N samples
- Detailed confusion matrix
- Per-class metrics (Precision, Recall, F1)
- Top-K error analysis

**Output Example:**
```
🔍 PRE-TRAINING MODEL EVALUATION
Model: Qwen/Qwen3-14B
Test samples: 998

📊 Baseline Accuracy: 45.23%
⏱️  Average inference: 1.850s/sample
```

### 3. Fine-tuning

Start the fine-tuning process:

```bash
python scripts/03_training.py
```

**Configuration Highlights:**
- **Epochs:** 2
- **Batch Size:** 16 per device
- **Gradient Accumulation:** 2 steps
- **Effective Batch Size:** 32
- **Learning Rate:** 2e-4 with cosine schedule
- **Precision:** BF16
- **LoRA Rank:** 64

**Training Output:**
```
🚀 Training başlıyor...
📊 Train samples: 4987
📊 Effective batch size: 32
📊 Steps/epoch (est): ~156
📊 Total steps (est): ~312

Epoch 1/2: 100%|██████████| 156/156 [12:34<00:00, 4.84s/it]
Eval Loss: 0.3245

💾 Best model saved!
✅ Training tamamlandı!
```

### 4. Post-training Evaluation

Evaluate the fine-tuned model:

```bash
python scripts/04_post_eval.py
```

Compare results with baseline:
```
📊 POST-TRAINING EVALUATION REPORT
Accuracy: 87.52% (↑42.29% from baseline)

[3] SINIF BAZLI METRİKLER
Label | Support | Precision |  Recall |     F1
    A |     203 |    85.21% |  88.17% | 86.67%
    B |     198 |    89.34% |  86.36% | 87.83%
    C |     201 |    88.56% |  89.55% | 89.05%
    D |     195 |    87.18% |  84.62% | 85.88%
    E |     201 |    86.57% |  89.05% | 87.79%
```

### 5. Single Example Testing

Test the model on a single example:

```bash
python scripts/05_single_test.py
```

Interactive testing with detailed analysis:
```python
test_ex = {
    "metin": "Paragraf metni...",
    "soru": "Soru?",
    "secenekler": ["A) ...", "B) ...", ...],
    "dogru_cevap": "D"
}
```

## ⚙️ Model Configuration

### Base Model
- **Model:** Qwen/Qwen3-14B
- **Parameters:** 14B
- **Quantization:** 4-bit (NF4)
- **Compute dtype:** BF16

### LoRA Configuration

```python
lora_config = LoraConfig(
    r=64,                    # Rank
    lora_alpha=128,          # Alpha (scaling)
    lora_dropout=0.1,        # Dropout
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
```

**Memory Efficiency:**
- Trainable params: ~118M (0.84% of total)
- Memory reduction: ~99.16%

## 🎛️ Training Configuration

### Optimization
- **Optimizer:** AdamW (Fused)
- **Learning Rate:** 2e-4
- **LR Schedule:** Cosine with 3% warmup
- **Weight Decay:** 0.01
- **Grad Clip:** 1.0

### Precision & Memory
- **Mixed Precision:** BF16
- **Gradient Checkpointing:** Enabled
- **TF32:** Enabled (A100)

### Data Loading
- **Workers:** 4
- **Pin Memory:** True
- **Prefetch Factor:** 2
- **Group by Length:** True

### Evaluation & Saving
- **Eval Strategy:** Every epoch
- **Save Strategy:** Every epoch
- **Save Total Limit:** 3 (keep best 3 checkpoints)
- **Load Best at End:** True
- **Metric:** Eval Loss (lower is better)

### Early Stopping
- **Patience:** 2 epochs
- **Threshold:** 0.0005

## 📊 Results

### Expected Performance

| Metric | Baseline (Pre-FT) | Fine-tuned | Improvement |
|--------|-------------------|------------|-------------|
| **Accuracy** | 40-50% | 85-92% | +40-45% |
| **Macro F1** | 38-48% | 83-90% | +40-45% |
| **None predictions** | 10-20% | <2% | -90% |

### Training Time

| Hardware | Samples | Time per Epoch | Total Time |
|----------|---------|----------------|------------|
| A100 80GB | 5000 | ~15-20 min | ~30-40 min (2 epochs) |
| V100 32GB | 5000 | ~25-35 min | ~50-70 min (2 epochs) |

### Confusion Matrix Example

```
True\Pred |       A |       B |       C |       D |       E |    None
------------------------------------------------------------------------
        A |     179 |       8 |       7 |       5 |       3 |       1
        B |       6 |     171 |       9 |       7 |       4 |       1
        C |       5 |       8 |     180 |       5 |       3 |       0
        D |       7 |       6 |       5 |     165 |      11 |       1
        E |       4 |       5 |       4 |       8 |     179 |       1
```

## 💻 Hardware Requirements

### Minimum Requirements
- **GPU:** 40GB VRAM (A100 40GB, A6000)
- **RAM:** 32GB
- **Storage:** 50GB free space

### Recommended Setup
- **GPU:** 80GB VRAM (A100 80GB)
- **RAM:** 64GB+
- **Storage:** 100GB+ SSD

### Cloud Options
- **Google Colab Pro+** (A100 available)
- **AWS:** p4d instances (A100)
- **Azure:** NC A100 v4 series
- **Lambda Labs:** A100 instances

### Cost Estimation
- **A100 80GB (Cloud):** ~$2-3/hour
- **Total training cost:** ~$2-4 (2 epochs, ~1-1.5 hours)

## 🔧 Troubleshooting

### Out of Memory (OOM)

**Solution 1:** Reduce batch size
```python
per_device_train_batch_size=8  # Instead of 16
gradient_accumulation_steps=4  # Instead of 2
```

**Solution 2:** Reduce max sequence length
```python
MAX_LEN = 384  # Instead of 512
```

**Solution 3:** Use 8-bit quantization
```python
load_in_8bit=True,
load_in_4bit=False,
```

### Slow Training

**Check:**
1. Enable gradient checkpointing
2. Use `dataloader_num_workers=4`
3. Enable TF32 (A100 only)
4. Verify mixed precision (BF16)

### Poor Performance

**Try:**
1. Increase training epochs (3-4)
2. Adjust learning rate (1e-4 to 5e-4)
3. Check data quality
4. Increase LoRA rank (128)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


---

⭐ If you find this project helpful, please consider giving it a star!

**Happy Fine-tuning! 🚀🇹🇷**
