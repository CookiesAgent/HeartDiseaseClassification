# Multi-Label ECG Classification with Deep Learning
This repository contains the implementation of a deep learning pipeline for multi-label ECG classification using the PTB-XL dataset. The project explores both time-series models (GRU) and image-based deep learning models (CNN) for detecting cardiovascular conditions from ECG signals.

The main goal is to investigate whether transforming ECG signals into image representations can help convolutional neural networks capture structural patterns that may not be easily detected in raw time-series signals.


# Dataset

This project uses the **PTB-XL ECG dataset**, a large-scale publicly available ECG dataset released on PhysioNet.

PhysioNet dataset page:  
[https://physionet.org/content/ptb-xl/](https://physionet.org/content/ptb-xl/1.0.1/)

Kaggle mirror:  
[https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data)

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

