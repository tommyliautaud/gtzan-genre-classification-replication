# 🎵 GTZAN Genre Classification — MLP and LSTM Replication Study

This repository reproduces and analyzes results from a published machine learning study on music genre classification using the [GTZAN dataset]([http://marsyas.info/downloads/datasets.html](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1244/final-projects/DominicJosephDeMarcoEricMartzReginaTHTa.pdf).  
It compares **MLP** (feed-forward) and **LSTM** (recurrent) architectures trained on **3-second, 6-second, and 9-second audio feature slices**.

---

## 🧠 Overview

The goal of this project is to replicate the performance reported in the paper:

> *“For our LSTM model, we achieved a very high test rating with 93% on the 3-second, 76% on the 6-second, and 73% on the 9-second time slices.”*

Using PyTorch, we trained comparable models from scratch, evaluated their accuracy, F1-scores, and confusion matrices, and investigated the effect of *leaky splits*, *sequence length*, and *regularization techniques* on generalization.

---


---

## ⚙️ Implementation Details

**Framework:** PyTorch  
**Dataset:** GTZAN (10 genres × 100 tracks = 1000 audio clips)  
**Features:** 58-dimensional audio features extracted in 3-second windows  
**Models:**  
- **MLP:** 2-layer fully connected network with dropout and ReLU  
- **LSTM:** 2-layer bidirectional LSTM with 64 hidden units and dense ReLU + dropout  
**Optimizer:** Adam (lr = 0.001 – 0.01)  
**Epochs:** 50 – 100  
**Batch size:** 16  
**Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix  

---

## 🧩 Experiment Design

We evaluate the models at two levels:

| Metric | Description |
|---------|--------------|
| **Window-level accuracy** | Measures how well the model classifies each 3-second audio slice. |
| **Clip-level accuracy** | Aggregates predictions across all slices of a song (majority vote) for more stable evaluation. |

This dual-level analysis helps identify whether the model is learning *local spectral patterns* or capturing *long-term musical structure*.

---

## 📊 Results Summary

| Model | Time Slice | Accuracy | Macro F1 |
|--------|-------------|-----------|----------|
| MLP | 30s | 67% | 0.67 |
| MLP | 3s | ~79% | 0.78 |
| LSTM | 3s | 75% | 0.75 |
| LSTM | 6s | 73% | 0.73 |
| LSTM | 9s | 73% | 0.73 |

---

## 🔍 Discussion

While the replicated models performed well, we were **unable to reach the reported 93% test accuracy** for the LSTM on 3-second features.  
We hypothesize that the original paper may have:
- Used **leaky data splits** (windows from the same song appearing in both train/test).  
- Applied **additional feature normalization or augmentation** not mentioned in detail.  
- Tuned **hidden layer sizes, dropout, or regularization** more extensively.  

We tested variations of these factors, but the best results stabilized around 75–78% accuracy — suggesting the paper’s results may have involved additional preprocessing or evaluation leakage.

---


