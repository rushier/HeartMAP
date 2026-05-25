# HeartMAP: Multi-Modal Heart Failure Assessment with Myocardial Late Gadolinium Enhancement Prediction

HeartMAP is a deep learning framework for predicting myocardial Late Gadolinium Enhancement (LGE) from multi-modal cardiac data including echocardiography videos and electrocardiogram (ECG) signals. The model leverages a Space-Time Factorized Vision Transformer (ViViT) architecture with ECG integration via HuBERT-ECG, and a Graph Neural Network (GNN) module for segmental allocation.

## Key Features

- **Multi-modal fusion**: Integrates echocardiography videos (multiple views: A4C, A3C, A2C, MID, MV) with 12-lead ECG signals
- **EchoPrime backbone**: Uses pretrained MViT v2 (EchoPrime) as the echocardiography encoder
- **HuBERT-ECG**: Self-supervised ECG representation learning for robust cardiac signal encoding
- **Graph Neural Network (GNN)**: Prior-guided GNN for 16-segment myocardial LGE prediction and segmental allocation
- **Multi-task learning**: Jointly predicts global LGE classification, segmental LGE scores, and total enhancement values
- **Space-Time Factorized ViViT**: Efficiently models spatial, temporal, and multi-view dependencies

## Directory Structure

```
HeartMAP/
├── run.py                          # Main training and inference entry point
├── requirements.txt                # Python dependencies
├── configs/
│   └── default.yml                 # Default configuration file
├── csv_files/                      # Data split CSV files
│   └── external/                   # External test cohort CSVs
├── models/
│   └── pretrained/                 # Pretrained model weights directory
├── checkpoints/                    # Trained model checkpoints
├── data/                           # Processed data directory
├── src/
│   ├── engine_joint.py             # Main training/testing engine
│   ├── builders/
│   │   ├── model_builder.py        # Model construction
│   │   ├── data_builder.py         # DataLoader construction
│   │   ├── criterion_builder.py    # Loss function construction
│   │   ├── optimizer_builder.py    # Optimizer construction
│   │   ├── scheduler_builder.py    # Learning rate scheduler
│   │   ├── transform_builder.py    # Data augmentation transforms
│   │   ├── evaluator_builder.py    # Evaluation metrics
│   │   ├── meter_builder.py        # Loss tracking meters
│   │   └── generator.py            # EchoPrime backbone (MViT v2)
│   ├── core/
│   │   ├── model_ECG_backbone.py   # ViViT + ECG fusion with EchoPrime backbone
│   │   ├── data.py                 # Dataset classes
│   │   ├── criterion.py            # Loss functions
│   │   ├── evaluators.py           # Evaluation metrics (AUC, F1, etc.)
│   │   ├── meters.py               # Average meters for loss tracking
│   │   ├── transformer.py          # Transformer encoder/decoder
│   │   ├── fine_grained_lge_value.py  # Prior-GNN for segmental LGE allocation
│   │   └── vit/                    # Vision Transformer implementation
│   │       ├── vit.py
│   │       ├── vit_configs.py
│   │       └── vit_utils.py
│   └── utils/
│       ├── misc.py                 # Miscellaneous utilities
│       └── vis.py                  # Visualization utilities
```

## Data Preparation

### Input Data

The model expects the following data for each patient:

1. **Echocardiography Videos**: Multi-view echo videos (A4C, A3C, A2C, MID, MV) stored as NPZ files
2. **12-Lead ECG Signals**: ECG recordings sampled at 500 Hz, stored as NumPy arrays
3. **CMR LGE Labels**: Binary LGE classification labels (0/1), 16-segment LGE scores, and total enhancement values

### Data CSV Format

Create a CSV file at `csv_files/input_LGE_mri_ecg_value.csv` with the following columns:

| Column | Description |
|--------|-------------|
| `PID` | Patient ID |
| `Echodate` | Echo examination date |
| `a4c`, `a3c`, `a2c`, `mid`, `mv` | Paths to NPZ files for each echo view |
| `ecg_path` | Path to 12-lead ECG NumPy file |
| `LGE` | Binary LGE classification label (0 or 1) |
| `LGE1` | 16-segment LGE segmental scores (comma-separated string) |
| `total_enhanced` | Total number of enhanced segments |
| `mri_path` | Path to CMR MRI data (optional) |

### Data Split CSVs

Place train/val/test split CSVs in `csv_files/`:
- `train_LGE_{seed}_relabel.csv`
- `val_LGE_{seed}_relabel.csv`
- `test_LGE_{seed}_relabel.csv`

Each split CSV should have matching `PID` and `Echodate` columns to identify patients.

### Pretrained Models

Download the following pretrained weights and place them in `models/pretrained/`:

| File | Description | Source |
|------|-------------|--------|
| `echo_prime_encoder.pt` | EchoPrime MViT v2 encoder | [EchoPrime](https://github.com/echoprime/EchoPrime) |
| `hubert_ecg_small.pt` | HuBERT-ECG pretrained model | [HuBERT-ECG](https://github.com/bakergilab/HuBERT-ECG) |
| `spacetime_unet_apical.pt` | SpaceTime UNet for apical views | (pretrained on echo segmentation) |
| `spacetime_unet_short.pt` | SpaceTime UNet for short-axis views | (pretrained on echo segmentation) |

### External Dependencies

The project requires the following external repositories:

1. **[HuBERT-ECG](https://github.com/bakergilab/HuBERT-ECG)**: Install via pip or clone into the project root:
   ```bash
   git clone https://github.com/bakergilab/HuBERT-ECG.git
   cd HuBERT-ECG-master/code
   pip install -e .
   ```

2. **EchoPrime**: The default backbone uses torchvision's standard `mvit_v2_s()` and requires no custom network files.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-org/HeartMAP.git
cd HeartMAP

# Install dependencies
pip install -r requirements.txt

# Install HuBERT-ECG
git clone https://github.com/bakergilab/HuBERT-ECG.git HuBERT-ECG-master
cd HuBERT-ECG-master/code
pip install -e .
cd ../..

# Download pretrained weights
mkdir -p models/pretrained
# Download echo_prime_encoder.pt, hubert_ecg_small.pt, etc.
```

## Usage

### Training

```bash
# Train the full model (EchoPrime backbone, prior-GNN fusion)
CUDA_VISIBLE_DEVICES=0,1 python run.py \
    --config_path configs/default.yml \
    --save_dir ./logs/experiment_1
```

### Testing / Inference

```bash
# Run inference on test set
CUDA_VISIBLE_DEVICES=0 python run.py \
    --config_path configs/default.yml \
    --save_dir ./logs/experiment_1 \
    --test
```

### Model Weights

The trained model checkpoint is available at:
```
checkpoints/checkpoint_best_loss.pth
```

To use the pretrained checkpoint, set the `checkpoint_path` in `configs/default.yml` or provide it via command line.

## Configuration

Key configuration options in `configs/default.yml`:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model.backbone_name` | Echo encoder backbone | `EchoPrime` |
| `model.fuse_mode` | Fusion strategy for multi-modal data | `prior_GNN` |
| `model.loc` | Enable segmental location prediction | `True` |
| `data.chamber_list` | Echo views to use | `a4c+a3c+a2c+mid+mv` |
| `data.add_ecg` | Enable ECG modality | `True` |
| `data.lge_value` | Enable LGE value regression | `True` |
| `train.epochs` | Number of training epochs | `60` |
| `train.batch_size` | Training batch size | `4` |
| `train.optimizer.lr` | Learning rate | `1e-5` |

## Model Architecture

HeartMAP employs a multi-modal fusion architecture:

1. **Spatial Encoder (STE)**: Vision Transformer processes each echo frame into patch embeddings
2. **Temporal Encoder (TTE)**: Transformer aggregates frame-level features across the cardiac cycle
3. **View Encoder (VTE)**: Transformer fuses multi-view information with ECG embeddings
4. **ECG Encoder**: HuBERT-ECG extracts features from 12-lead ECG signals, concatenated before VTE
5. **GNN Module**: Prior-guided graph neural network performs segmental allocation from global LGE prediction to 16 AHA myocardial segments
6. **Output Heads**: Multi-task heads for classification, regression, and segmental prediction

## Citation

If you use HeartMAP in your research, please cite:

```bibtex
@article{heartmap2024,
  title={HeartMAP: Multi-Modal Heart Failure Assessment with Myocardial Late Gadolinium Enhancement Prediction},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This project is for research purposes only. See [LICENSE](LICENSE) for details.

## Acknowledgements

- [EchoPrime](https://github.com/echoprime/EchoPrime) for the echocardiography video encoder
- [HuBERT-ECG](https://github.com/bakergilab/HuBERT-ECG) for the ECG self-supervised learning model
- [VideoMAE](https://github.com/MCG-NJU/VideoMAE) for video masked autoencoders
- [ViViT](https://arxiv.org/abs/2103.15691) for the space-time factorized video transformer
