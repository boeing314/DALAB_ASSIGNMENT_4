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

## Environment & Dependencies
Recommended: Python 3.9+ and the following packages.

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install directly:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter
```

Optional (for faster GMM / plotting):
```bash
pip install joblib tqdm
```

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

## What to look for (expected outputs)
- Class distribution printout (heavily imbalanced).
- Baseline metrics: low recall on minority class; precision may be high.
- GMM + CBU rebalanced results: improved Recall and F1 for the minority class (check the notebook’s summary table and bar chart).

---

## Notes & Tips
- Use AIC/BIC plots in the notebook to select optimal GMM component count.
- When sampling from GMM, ensure generated samples are within realistic ranges (feature scaling inverse transform if required).
- Keep the test set untouched and imbalanced — evaluation must reflect the original distribution.

---

## Author / Contact
Asher — reach out in the notebook comments or add a note at the top of the notebook for clarifications.

---

## License
Use for coursework and personal learning only.

