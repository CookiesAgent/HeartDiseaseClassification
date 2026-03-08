# Multi-Label ECG Classification with Deep Learning
This repository contains the implementation of a deep learning pipeline for multi-label ECG classification using the PTB-XL dataset. The project explores both time-series models (GRU) and image-based deep learning models (CNN) for detecting cardiovascular conditions from ECG signals.

The main goal is to investigate whether transforming ECG signals into image representations can help convolutional neural networks capture structural patterns that may not be easily detected in raw time-series signals.


# Dataset

This project uses the **PTB-XL ECG dataset**, a large-scale publicly available ECG dataset released on PhysioNet.

PhysioNet dataset page:  
[https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/1.0.1/)

Kaggle mirror:  
[https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data)

The dataset contains more than 21,000 ECG recordings collected from nearly 19,000 patients and annotated by cardiologists.
Due to size limitations, the dataset is not included in this repository.

# Classification Task

The project focuses on detecting the following diagnostic classes:

| Label | Description |
|------|-------------|
| MI | Myocardial Infarction |
| STTC | ST/T Change |
| CD | Conduction Disturbance |
| HYP | Hypertrophy |
| AD | Absence of disease (normal ECG) |

Because ECG recordings may contain multiple abnormalities, this is a **multi-label classification problem**.


All experiments, preprocessing, training, and evaluation are implemented inside the **Jupyter notebooks** located in the `notebooks/` directory.

---

# Installation

## Clone the repository

```bash
git clone https://github.com/<username>/HeartDiseaseClassification.git
cd HeartDiseaseClassification
```

# Dependencies 
Dependencies

Main libraries used in this project:

- Python 3.10+
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- WFDB
- tqdm
- PyYAML
- pytest


# Workflow
## Baseline Model
Notebook: notebooks/HeartDiseaseClassification_baseline.ipynb

This notebook implements the **GRU baseline model** that processes ECG signals directly as time-series data.

Model configuration:
- Two stacked GRU layers
- Hidden dimension: 128
- Dropout: 0.3
- Optimizer: Adam
- Batch size: 256

---

## CNN with Attention
Notebook: notebooks/AlexNetAtt_Notebook.ipynb
This notebook implements an **AlexNet-based CNN with self-attention layers** trained on ECG image representations.

Signal transformations:

- Gramian Angular Field (GAF)
- Recurrence Plot (RP)
- Markov Transition Field (MTF)

Each ECG record becomes a **9-channel image (3 leads × 3 transformations)**.

---

## Residual CNN Experiment

Notebook: notebooks/ResidualCNN_SOTA.ipynb

This notebook explores a **Residual CNN architecture** as a potential improvement over the baseline models.

---

# Preprocessing

The preprocessing pipeline includes:

1. Bandpass filtering (1–45 Hz Butterworth filter)
2. Z-score normalization per ECG record
3. Selection of ECG leads:
  -- Lead I
  -- Lead II
  -- Lead V2
# Evaluation Metrics

The models are evaluated using:

- Sensitivity (Recall)
- Specificity
- Accuracy
- F1 Score

Because this is a **multi-label classification task**, metrics are computed per class and averaged.

Sensitivity is emphasized since missing a disease (false negative) can have serious clinical consequences.

---

# Results

Model performance:

| Model | Sensitivity | Specificity | Accuracy | F1 Score |
|------|------|------|------|------|
| GRU Baseline | 0.7534 | 0.8129 | 0.8001 | 0.6048 |
| AlexNet + Attention | 0.7448 | 0.7651 | 0.7614 | 0.5915 |

Generated outputs include:

- training curves
- confusion matrices
- evaluation metrics
- XAI visualizations
 They can be found for each model seperately in their respective notebooks.

# Running the Experiments
Launch Jupyter Notebook:
Run the notebooks in this order:

1. `HeartDiseaseClassification_baseline.ipynb`
2. `AlexNetAtt_Notebook.ipynb`
3. `ResidualCNN_SOTA.ipynb`

Each notebook contains the full workflow for preprocessing, training, and evaluation.

# Authors

Alimbek Alimbekov — Visualization and analytics  
Aruzhan Nurmanova — CNN model training  
Lailim-Adina Kozhamkulova — Data preprocessing  
Baurzhan Kumarioldanov — Baseline model development  
Dulat Rakhymkul — Baseline model training
