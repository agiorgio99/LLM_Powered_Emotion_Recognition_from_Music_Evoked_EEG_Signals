# LLM-Powered Emotion Recognition from Music-Evoked EEG Signals

**Master's Thesis in Artificial Intelligence and Robotics** *Sapienza University of Rome* **Author:** Antonello Giorgio  
**Advisor:** Prof. Danilo Comminiello  
**Year:** 2024/2025

---

## 📖 Abstract

Emotion-aware technology is transforming fields ranging from adaptive gaming to mental health monitoring. However, reliably decoding human emotions from brain signals (EEG) remains a challenge due to high dimensionality, low signal-to-noise ratio, and strong inter-subject variability.

This repository contains the implementation of my Master's Thesis, which proposes a **dual-branch neural architecture** combining the temporal sensitivity of **Large Language Models (LLMs)** with the spatial reasoning of **Dynamic Graph Convolutional Neural Networks (DGCNNs)**.

By leveraging **Low-Rank Adaptation (LoRA)** and a novel **masked embedding reconstruction loss**, the model fine-tunes foundation models (GPT-2, LLaMA) directly on EEG data, achieving State-of-the-Art (SOTA) performance on the DEAP dataset.

---

## 🚀 Key Features

* **Dual-Branch Architecture**:
    * **Temporal Branch**: An EEG-Transformer encoder coupled with an autoregressive LLM decoder (GPT-2 / LLaMA) to reconstruct masked signal windows.
    * **Spatial Branch**: A Dynamic Graph CNN (DGCNN) that learns inter-electrode functional connectivity.
* **LLM Fine-Tuning**: First known application of direct fine-tuning (via LoRA) of LLMs on the DEAP dataset for joint arousal/valence classification.
* **Self-Supervised Learning**: Uses a reconstruction loss ($\mathcal{L}_{rec}$) to regularize temporal dynamics and improve generalization in low-data regimes.
* **Fusion Strategy**: Implements confidence-aware gating to dynamically prioritize the most reliable prediction stream (Temporal vs. Spatial).

---

## 🧠 Model Architecture

The system processes EEG segments (from the DEAP dataset) through two parallel paths:

1.  **Temporal Encoder**: Splits signals into chunks, processes them via a Transformer Encoder, and uses an LLM to predict masked latent embeddings.
2.  **Spatial Encoder**: Treats electrodes as graph nodes and learns dynamic adjacency matrices to capture brain topology.

The outputs are fused using a weighted average or confidence-based selection to produce the final binary classification for **Arousal** and **Valence**.

---

## 📊 Results

The model was evaluated on the **DEAP dataset** (32 subjects) using a subject-dependent protocol. It achieved new State-of-the-Art results, outperforming baselines like *TSCeption*, *LGGNet*, and *MT-LGSGCN*.

| Method | Arousal (Acc) | Arousal (F1) | Valence (Acc) | Valence (F1) |
| :--- | :---: | :---: | :---: | :---: |
| SVM | 60.37% | 57.33% | 55.19% | 57.87% |
| TSCeption | 61.57% | 63.24% | 59.14% | 62.33% |
| LGGNet-Gen | 61.81% | 64.49% | 59.14% | 64.58% |
| MT-LGSGCN | 63.59% | 65.11% | 61.69% | 65.23% |
| **Ours (LLM-Powered)** | **66.99%** | **65.78%** | **64.69%** | **64.53%** |

*Ablation studies confirmed that the fine-tuned GPT-2 backbone with LoRA yielded the best efficiency-performance trade-off.*

---

## 🛠️ Technologies Used

* **Python** & **PyTorch**
* **HuggingFace Transformers** (GPT-2, LLaMA 3.1/3.2)
* **PEFT (LoRA)** for efficient fine-tuning
* **PyTorch Geometric / Graph Layers** for DGCNN
* **DEAP Dataset** (EEG Signal Processing)

---

## ⚠️ Resources & Weights

Due to the size of the fine-tuned models and licensing restrictions of the dataset:

* **Model Weights:** The LoRA adapters and checkpoints are not hosted in this repository.
* **Dataset:** You must obtain access to the DEAP dataset directly from the [official website](http://www.eecs.qmul.ac.uk/mmv/datasets/deap/).

> **Note:** If you are a researcher interested in the model weights, the pre-processed data pipeline, or specific implementation details for reproducibility, please **contact me privately** via email or LinkedIn (links in my GitHub profile).

---

## 📄 Citation

If you find this work useful, please refer to the thesis:

> **Giorgio, A. (2025).** *LLM-Powered Emotion Recognition from Music-Evoked EEG Signals*. Master's Thesis in AI & Robotics, Sapienza University of Rome.
