# DA5401 A4 — GMM-Based Synthetic Sampling for Imbalanced Data

## Overview
This repository contains the work for **DA5401 Assignment 4**: using a Gaussian Mixture Model (GMM) to generate synthetic samples for the minority class (fraud) and evaluating its impact on classifier performance versus a baseline logistic regression trained on imbalanced data.

**Objective:** Build a reproducible pipeline that:
1. Trains a baseline Logistic Regression on the imbalanced dataset.
2. Fits a GMM to minority-class samples and generates synthetic minority samples.
3. Applies clustering-based undersampling (CBU) to the majority class.
4. Trains a classifier on the rebalanced dataset and compares metrics (Precision, Recall, F1) on the original imbalanced test set.

---

## Files
- `asgn4.ipynb` — Jupyter Notebook with code, visualizations, and narrative (uploaded).
- `DA5401 A4 GMM.pdf` — Assignment brief and instructions (uploaded).
- `README_DA5401_A4.md` — This file.

---

## Dataset
Use the Kaggle Credit Card Fraud dataset:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it in the same folder as the notebook (or update the path in the notebook).

---

## How to run (quick)
1. Open the notebook:
```bash
jupyter notebook asgn4.ipynb
```
2. Cell order / recommended run:
   - Start with the **Data Loading & Analysis** section (Part A).
   - Proceed to **Baseline Model** cells (train baseline, evaluate on test set).
   - Run **GMM training & synthetic generation** (Part B).
   - Run **CBU undersampling** and combine generated minority samples.
   - Train the **rebalanced model** and run evaluation (Part C).
3. All random seeds are fixed in the notebook for reproducibility (`random_state=42`).

---

## Author 
Asher 
---


