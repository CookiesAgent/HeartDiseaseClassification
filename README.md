# Multi-Label ECG Classification with Deep Learning
TThis repository provides a systematic deep learning benchmark for multi-label ECG classification. The project evaluates five diverse architectures across two major datasets (PTB-XL and Georgia) to detect four cardiac superclasses: Myocardial Infarction (MI), ST/T-wave changes (STTC), Conduction Disturbance (CD), and Hypertrophy (HYP).

Our research investigates three input representation paradigms:Raw 1D Time-Series: Directly processing the 12-lead ECG signal.2D STFT Spectrograms: Transforming signals into time-frequency representations.Hybrid/Fusion Approaches: Combining temporal (1D) and spectral (2D) features for enhanced performance


# Dataset

We utilize two publicly available 12-lead ECG datasets, resampled to **100 Hz** for consistency:

* **PTB-XL**: 21,837 records (10s duration). 

  * PhysioNet dataset page:  [https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/1.0.1/)

  * Kaggle mirror:  [https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data)
* **Georgia 12-Lead ECG Challenge**: 10,344 records.
  * Kaggle link: [https://www.kaggle.com/datasets/physionet/georgia-12lead-ecg-challenge-database](https://www.kaggle.com/datasets/physionet/georgia-12lead-ecg-challenge-database)


Due to their size, the datasets is not included in this repository.

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

*  Clone the repository

```bash
git clone https://github.com/CookiesAgent/HeartDiseaseClassification.git
cd HeartDiseaseClassification
pip install -r requirements.txt
```

* Preprocessing:
Run the notebooks in data_setup/ to prepare the datasets.

* Training:
Navigate to the respective dataset folder in Notebooks/ and run the model training notebooks (e.g., ECG_CRNN_PTB_XL.ipynb).


* Explainability:
Use the evaluation notebooks to generate Grad-CAM and Integrated Gradients visualizations, which highlight clinically relevant features like R-peaks and ST-segments


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

## Preprocessing
Preprocessing Pipeline:
- Bandpass filtering (1–45 Hz).
- Z-score normalization.
- NORM undersampling to address class imbalance.
- Multi-label stratified 70/15/15 data split.

You can download and preprocess the PTB_XL dataset by running the notebook:
`data_setup/PTB_XL_dataset.ipynb`

For working with Georgia dataset, run the notebook:
`data_setup/Georgia_Preprocessing.ipynb`

## Model Architectures

The project investigates three input representation paradigms:

| Model | Input Type | Description | Key Components |
| :--- | :--- | :--- | :--- |
| **Baseline GRU** | Raw 1D | Standard temporal model | 2-layer stacked GRU, Hidden Dim: 128 |
| **ECG-CRNN** | Raw 1D | Convolutional Recurrent NN | SE-ResNet + BiLSTM |
| **ResNet2D-ECG** | 2D STFT | Spectral-only model | ResNet-18 on 12-channel spectrograms |
| **ResNet18-BiLSTM**| 2D STFT | Hybrid Spectral-Temporal | Pretrained ResNet-18 + BiLSTM + SE Attention |
| **Fusion GRU+2DCNN**| 1D + 2D | **Late-Fusion (SOTA)** | Parallel BiGRU (1D) and 2D CNN-CBAM branches |
| **Transformer-Hybrid**| Raw 1D | Attention-based model | 1D-CNN + BiLSTM + Transformer Encoder |

## Evaluation Metrics

The models are evaluated using:

- Sensitivity (Recall)
- Specificity
- Accuracy
- F1 Score

Because this is a **multi-label classification task**, metrics are computed per class and averaged.

Sensitivity is emphasized since missing a disease (false negative) can have serious clinical consequences.

---

## Experimental Results

All models were evaluated using Macro-averaged metrics after per-class threshold tuning on the validation set.

### PTB-XL Dataset Results
| Model | Macro-F1 | Macro-AUC | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **Fusion GRU+2DCNN** | **0.725** | **0.904** | 0.812 | 0.885 |
| Transformer-Hybrid | 0.714 | 0.884 | 0.795 | 0.864 |
| ResNet18-BiLSTM | 0.698 | 0.872 | 0.764 | 0.851 |
| ECG-CRNN | 0.672 | 0.848 | 0.741 | 0.832 |
| **Baseline GRU** | 0.605 | 0.821 | 0.753 | 0.813 |

### Georgia Dataset Results
| Model | Macro-F1 | Macro-AUC | Sensitivity | Specificity |
| :--- | :---: | :---: | :---: | :---: |
| **ECG-CRNN** | **0.732** | 0.838 | 0.801 | 0.824 |
| **Fusion GRU+2DCNN** | 0.710 | **0.883** | 0.788 | 0.871 |
| Transformer-Hybrid | 0.695 | 0.852 | 0.755 | 0.844 |
| ResNet2D-ECG | 0.654 | 0.811 | 0.722 | 0.798 |


Generated outputs also include:

- training curves
- confusion matrices
- evaluation metrics
- XAI visualizations
 They can be found for each model seperately in their respective notebooks as well as in Figures folder.


Each notebook contains the full workflow for training and evaluation.

# Authors

Dulat Rakhymkul - Target model training & tuning (ResNet2D, Fusion, LSTM Transfer)

Baurzhan Kumarioldanov - Baseline model development & Transformer-Hybrid architecture

Aruzhan Nurmanova - Baseline ECG-CRNN training & tuning

Alimbek Alimbekov - Visualization, analytics & explainability analysis (XAI)

Lailim-Adina Kozhamkulova - Georgia ECG preprocessing pipeline & stratified splitting


