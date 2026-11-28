# SADKD: Structure-Aware and Discriminative Knowledge Distillation for Anomaly Detection

## Overview

**SADKD** (Structure-Aware and Discriminative Knowledge Distillation) is an unsupervised anomaly detection framework designed for industrial defect inspection. It aims to improve structural feature representation and anomaly discrimination by extending conventional reverse knowledge distillation with pseudo-anomaly-guided restoration and feature repulsion.

Unlike conventional knowledge distillation methods that primarily reconstruct normal feature representations, SADKD introduces a discriminative learning strategy to explicitly distinguish normal and anomalous representations.

The proposed framework consists of three key components:

- **Residual Coordinate Attention Refinement (R-CAR):** Captures long-range spatial dependencies along the height and width directions to enhance structural feature representations while preserving spatial information.
- **Soft Pseudo-Anomaly (SPA) Generation:** Generates pseudo-anomalous samples through feature perturbations and facilitates self-supervised restoration of normal representations.
- **Repulsive Feature Distillation (RFD):** Combines normal feature alignment, adaptive focusing, and anomaly repulsion to improve feature discrimination and anomaly localization.

SADKD is evaluated on two widely used industrial anomaly detection benchmarks: **MVTec AD** and **VisA**.

## Method Architecture

The overall architecture of SADKD is illustrated below.

![SADKD Architecture](figures/SADKD_framework.png)

### Main Components

**1. Residual Coordinate Attention Refinement (R-CAR)**

- Aggregates spatial information along horizontal and vertical directions.
- Captures long-range dependencies while preserving positional information.
- Enhances structure-sensitive feature representations through residual refinement.

**2. Soft Pseudo-Anomaly (SPA) Generation**

- Generates pseudo-anomalous samples from normal training images.
- Introduces spatially smoothed perturbations to simulate anomalous regions.
- Provides self-supervised training signals for normal feature restoration.

**3. Repulsive Feature Distillation (RFD)**

- Aligns reconstructed student features with clean teacher features.
- Introduces adaptive focusing to emphasize difficult reconstruction regions.
- Applies anomaly repulsion to distinguish normal and pseudo-anomalous representations.

## Quick Start

### Environment Configuration

**1. Create a virtual environment (optional):**

```bash
conda create -n sadkd python=3.10
conda activate sadkd
```

**2. Install dependencies:**

Install the required packages:

```bash
pip install -r requirements.txt
```

The main dependencies include:

- Python
- PyTorch
- Torchvision
- NumPy
- SciPy
- OpenCV
- Pillow
- Scikit-learn
- Matplotlib
- tqdm

Please refer to the actual implementation and `requirements.txt` for the exact package versions.

### Data Preparation

SADKD is evaluated on the following industrial anomaly detection datasets.

**1. MVTec AD Dataset**

- Official website: https://www.mvtec.com/company/research/datasets/mvtec-ad
- Contains 15 object and texture categories.
- Provides normal training images and normal/anomalous test images with pixel-level annotations.

**2. VisA Dataset**

- Official repository: https://github.com/amazon-science/spot-diff
- Contains 12 industrial object categories.
- Includes complex backgrounds, multiple objects, reflective surfaces, and diverse defect patterns.

**3. Directory Structure Example**

```text
/path/to/datasets/
├── MVTec_AD/
│   ├── bottle/
│   ├── cable/
│   ├── capsule/
│   ├── carpet/
│   └── ...
│
└── VisA/
    ├── candle/
    ├── capsules/
    ├── cashew/
    └── ...
```

Please organize the datasets according to the directory structure expected by the data-loading implementation.

## Training Execution

### 1. Configuration

Before training, configure the dataset path and experimental settings in the corresponding configuration file or training script.

The main parameters include:

```python
root_dir = "path/to/datasets"
dataset_name = "MVTec_AD"

image_size = 256
batch_size = 16
epochs = 200

learning_rate = 0.001
optimizer = "Adam"
scheduler = "CosineAnnealingLR"
```

### 2. Start Training

Run the training script:

```bash
python main.py
```

The training process uses normal training images and generated pseudo-anomalous samples to optimize the proposed SADKD framework.

The teacher encoder remains frozen, while the student reconstruction path is optimized.

## Inference and Testing

After training, evaluate the model on the corresponding test dataset.

Run:

```bash
python test.py
```

The evaluation process generates anomaly maps by measuring the discrepancy between teacher features and reconstructed student features.

The predicted anomaly maps are used for image-level anomaly detection and pixel-level anomaly localization.

**Note:** The commands above assume that `main.py` and `test.py` are the actual training and evaluation entry points. Please adjust the commands according to the implemented arguments and configuration files.

## Training Strategy

### Feature Learning

SADKD adopts a teacher-student architecture:

- The teacher encoder extracts reference features and remains frozen.
- The student reconstruction path consists of bottleneck transformation, R-CAR refinement, and decoder reconstruction.
- SPA generates pseudo-anomalous samples to construct self-supervised restoration tasks.
- RFD encourages alignment with clean teacher representations while separating anomalous representations.

### Optimization

The training strategy uses:

- Optimizer: Adam
- Initial learning rate: 0.001
- Batch size: 16
- Training epochs: 200
- Learning rate schedule: Cosine Annealing

The student network is trained using normal images and their pseudo-anomalous counterparts.

## Evaluation Metrics

SADKD is evaluated using standard industrial anomaly detection metrics:

- **Image-level AUROC (I-AUROC):** Measures image-level anomaly detection performance.
- **Pixel-level AUROC (P-AUROC):** Evaluates pixel-level anomaly localization accuracy.
- **AUPRO:** Measures region-based anomaly localization performance.

## Experimental Results

### 1. MVTec AD Dataset

SADKD achieves competitive performance on the MVTec AD benchmark.

The reported average results are:

| Metric | SADKD |
|:---|---:|
| I-AUROC | 99.31% |
| P-AUROC | 98.13% |
| AUPRO | 95.01% |

SADKD achieves 100% image-level AUROC on 8 out of 15 categories.

### 2. VisA Dataset

Experiments on VisA demonstrate the effectiveness of SADKD in handling complex industrial structures, reflective surfaces, and multi-instance objects.

SADKD achieves improved pixel-level anomaly localization performance compared with the evaluated baseline methods.

### 3. Ablation Study

Ablation experiments investigate the contributions of the three proposed components:

- R-CAR: Enhances structural feature representation.
- SPA: Improves self-supervised anomaly restoration.
- RFD: Strengthens the discrimination between normal and anomalous features.

These components jointly improve anomaly detection and localization performance.

## Visualization

SADKD produces anomaly localization maps during inference.

The visualization results can include:

- Original test images.
- Ground-truth anomaly masks.
- Predicted anomaly maps.
- Pixel-level anomaly localization results.

Example visualization:

![SADKD Visualization](figures/SADKD_visualization.png)

Please replace the example image paths with the actual visualization files generated by the implementation.

## Customization and Extensions

### 1. Modifying R-CAR

The R-CAR module can be adapted to different feature dimensions and backbone architectures to improve structural representation.

### 2. Adjusting SPA

The pseudo-anomaly generation strategy can be modified by changing perturbation regions, smoothing operations, and blending parameters.

### 3. Adjusting RFD

The RFD loss can be further investigated by adjusting the weights of reconstruction, adaptive focusing, and anomaly repulsion terms.
