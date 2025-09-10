# CLEAR: Contrastive Learning-based Embeddings for Attention-based Radiology

This repository contains the code for the MICCAI paper [**"Abnormality-Driven Representation Learning for Radiology Imaging"**](https://papers.miccai.org/miccai-2025/paper/4381_paper.pdf) by Marta Ligero, Tim Lenz, Georg Wölflein, Omar S.M. El Nahhas, Daniel Truhn, and Jakob Nikolas Kather.

## Abstract

Radiology deep learning pipelines predominantly employ end-to-end 3D networks based on models pre-trained on other tasks, which are then fine-tuned on the task at hand. In contrast, adjacent medical fields such as pathology have effectively adopted task-agnostic foundational models based on self-supervised learning (SSL), combined with weakly-supervised deep learning (DL). However, the field of radiology still lacks task-agnostic representation models due to the computational and data demands of 3D imaging and the anatomical complexity inherent to radiology scans. 

To address this gap, we propose **CLEAR**, a framework for 3D radiology images that uses extracted embeddings from 2D slices along with attention-based aggregation to efficiently predict clinical endpoints. As part of this framework, we introduce **LeCL** (Lesion-enhanced Contrastive Learning), a novel approach to obtain visual representations driven by abnormalities in 2D axial slices across different locations of the CT scans.

Specifically, we trained single-domain contrastive learning approaches using three different architectures: Vision Transformers, Vision State Space Models and Gated Convolutional Neural Networks. We evaluate our approach across three clinical tasks: tumor lesion location, lung disease detection, and patient staging, benchmarking against four state-of-the-art foundation models, including BiomedCLIP. Our findings demonstrate that Clear, using representations learned through Lecl, outperforms existing foundation models, while being substantially more compute- and data-efficient.

## Overview

Radiology deep learning pipelines often rely on computationally expensive 3D models pre-trained on unrelated tasks. CLEAR introduces a novel framework for 3D radiology imaging that leverages 2D slice embeddings and attention-based aggregation to efficiently predict clinical endpoints. The framework includes **Lesion-enhanced Contrastive Learning (LeCL)**, a semi-supervised method for learning visual representations driven by abnormalities in 2D axial slices.

### Key Contributions
1. **CLEAR Framework**: A domain-specific framework for radiology that integrates foundation models and attention-based methods for diverse clinical applications.
2. **LeCL**: A novel semi-supervised contrastive learning approach for abnormality-driven representation learning.
3. **Model Architectures**: Evaluation of Vision Transformers (ViT), Vision State Space Models (VSSM), and gated Convolutional Neural Networks (CNN) for domain-specific representation learning.
4. **Clinical Tasks**: Benchmarking across three tasks:
   - Tumor lesion location
   - Lung disease detection
   - Patient staging

### Results
CLEAR, using representations learned through LeCL, outperforms state-of-the-art foundation models (e.g., BiomedCLIP) while being significantly more compute- and data-efficient.

## Repository Structure
- **`/src`**: Core implementation of CLEAR and LeCL.
  - **`/LECL`**: Lesion-enhanced Contrastive Learning implementation
  - **`/extract_features`**: Feature extraction scripts for different models
  - **`/downstreamtasks`**: Downstream task evaluation scripts
- **`/models`**: Pre-trained models and architecture definitions.
- **`/data`**: Scripts for data preprocessing and loading.
- **`/experiments`**: Scripts for running experiments and evaluations.
- **`/notebooks`**: Jupyter notebooks for visualization and analysis.

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 2.0+

### Setup Environment
To set up the environment, use the provided `pyproject.toml` file. Install dependencies with:

```bash
uv sync
```

### Additional Dependencies
Install the required external repositories:

```bash
# VMamba (Vision State Space Models)
git clone https://github.com/MzeroMiko/VMamba.git src/VMamba

# MambaOut 
git clone https://github.com/yuweihao/MambaOut.git src/MambaOut
```

## Usage

### 1. Training LeCL Models

The main script for training LeCL (Lesion-enhanced Contrastive Learning) models is located at `src/LECL/main_moco.py`.

#### Basic Training Command

```bash
cd src/LECL
python main_moco.py \
    --data /path/to/your/training/data \
    --df-path /path/to/lesion/annotations.csv \
    --arch vit_conv_base \
    --batch-size 256 \
    --lr 1e-4 \
    --epochs 100 \
    --sampling Lesion \
    --multiprocessing-distributed \
    --dist-url 'tcp://localhost:23457' \
    --world-size 1 \
    --rank 0
```

#### Architecture Options
- `vit_small`, `vit_base`, `vit_conv_small`, `vit_conv_base`: Vision Transformers
- `vmamba`: Vision State Space Models (VSSM)
- `mambaout`: MambaOut architecture
- `swin`, `swin_base`: Other vision architectures

#### Key Parameters
- `--data`: Path to training images directory
- `--df-path`: Path to CSV file containing lesion annotations
- `--arch`: Model architecture (see options above)
- `--sampling`: Sampling strategy (`Lesion` for LeCL, `None` for standard MoCo)
- `--sample-weight`: Weight for lesion samples in contrastive loss
- `--batch-size`: Batch size (adjust based on GPU memory)
- `--epochs`: Number of training epochs

#### Example for Different Architectures

**Vision Transformer:**
```bash
python main_moco.py --data /data/images --df-path /data/lesions.csv --arch vit_conv_base --sampling Lesion
```

**VMamba (Vision State Space Model):**
```bash
python main_moco.py --data /data/images --df-path /data/lesions.csv --arch vmamba --sampling Lesion
```

**MambaOut:**
```bash
python main_moco.py --data /data/images --df-path /data/lesions.csv --arch mambaout --sampling Lesion
```

### 2. Feature Extraction

After training, extract features from your trained models using the scripts in `src/extract_features/SSL_models/`.

#### Extract Features with MambaOut

```bash
cd src/extract_features/SSL_models
python sslfeatures_extract_mambaout.py \
    --dataset_path /path/to/test/images \
    --output_dir /path/to/output/features \
    --model_path /path/to/trained/model.pth
```

#### Parameters
- `--dataset_path`: Directory containing images organized by patient/study
- `--output_dir`: Directory to save extracted features (.h5 files)
- `--model_path`: Path to your trained LeCL model checkpoint

### 3. Downstream Task Training

Train attention-based models for clinical tasks using the extracted features.

#### Cross-Validation Training

```bash
cd src/downstreamtasks
python train_crossval.py --features mambaout_features
```

#### Configuration
Edit the configuration in `utils/config_DL.py` to specify:
- Feature paths
- Clinical CSV files
- Model architectures (ABMIL, PMA, etc.)
- Training parameters

### 4. Data Format Requirements

#### Training Data Structure
```
training_data/
├── patient_001/
│   ├── slice_001.png
│   ├── slice_002.png
│   └── ...
├── patient_002/
│   └── ...
```

#### Lesion Annotation CSV
Required columns for LeCL training:
```csv
image,lesion_present,patient_id
patient_001_slice_001.png,1,patient_001
patient_001_slice_002.png,0,patient_001
patient_002_slice_001.png,1,patient_002
```

#### Clinical Data CSV
For downstream tasks:
```csv
image,label1,label2,all
patient_001,0,1,1
patient_002,1,0,1
```

## Example Workflows

### Complete Training Pipeline

1. **Prepare your data** according to the format above
2. **Train LeCL model:**
   ```bash
   python src/LECL/main_moco.py --data /data/train --df-path /data/lesions.csv --arch vit_conv_base --sampling Lesion --epochs 100
   ```
3. **Extract features:**
   ```bash
   python src/extract_features/SSL_models/sslfeatures_extract_mambaout.py --dataset_path /data/test --output_dir /features --model_path /models/checkpoint.pth
   ```
4. **Train downstream model:**
   ```bash
   python src/downstreamtasks/train_crossval.py --features extracted_features
   ```

### Quick Start for Testing

1. **Use pre-trained weights** (if available)
2. **Extract features** on a small dataset
3. **Run downstream training** with reduced epochs for validation

## Model Checkpoints

Trained models are saved with the following naming convention:
```
{architecture}-moco-lecl-1-{experiment_name}-checkpoint_{epoch:04d}.pth.tar
```

## Performance Tips

- **Multi-GPU Training**: Use `--multiprocessing-distributed` for faster training
- **Batch Size**: Adjust based on GPU memory (256-4096 recommended)
- **Learning Rate**: Scale with batch size (lr = base_lr * batch_size / 256)
- **Sampling Strategy**: Use `--sampling Lesion` for abnormality-driven learning

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient accumulation
2. **Missing dependencies**: Ensure VMamba and MambaOut are properly installed
3. **Data loading errors**: Check file paths and CSV format
4. **Distributed training**: Verify port availability and GPU configuration

### Debugging

Enable verbose logging by setting:
```bash
export PYTHONPATH="${PYTHONPATH}:./src"
export CUDA_LAUNCH_BLOCKING=1  # For debugging CUDA errors
```

## Citation

If you use this code, please cite our paper:
```bibtex
@InProceedings{Ligero_2025_MICCAI,
author = { Ligero*, Marta and Lenz*, Tim and W{\"o}lflein, Georg and El Nahhas, Omar S. M. and Truhn, Daniel and Kather, Jakob Nikolas},
title = { { Abnormality-Driven Representation Learning for Radiology Imaging } },
booktitle = {proceedings of Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
year = {2025},
publisher = {Springer Nature Switzerland},
volume = {LNCS 15963},
month = {September},
}
```

## Contact

For questions about the code or paper, please contact:
- Jakob Nikolas Kather: kather.jn@tu-dresden.de
- Open an issue on GitHub for technical problems

