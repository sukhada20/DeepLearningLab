# DL_4 – Regularization Techniques for Deep Learning

## Problem Statement
Implement L1/L2 Regularization and Dropout to reduce overfitting in a Housing Price Prediction task, and compare their performance against a baseline model.

## Activities Performed
- Trained a baseline DFFN model
- Applied L2 regularization to the model's weights
- Applied Dropout to the model architecture
- Compared MSE/MAE and training vs. validation loss curves across all three models

## Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- Matplotlib, NumPy

## Dataset
California Housing dataset (`sklearn.datasets.fetch_california_housing`).

## Workflow
1. Loaded the California Housing dataset
2. Performed an 80/20 train-test split
3. Standardized features using `StandardScaler`
4. **Baseline Model:** `Dense(64, relu) → Dense(32, relu) → Dense(1)`, Adam optimizer (lr=0.001), MSE loss
5. **L2 Regularization Model:** Same architecture with `kernel_regularizer=l2(0.001)` on hidden layers
6. **Dropout Model:** `Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.3) → Dense(1)`
7. Each model trained for 50 epochs (20% validation split)
8. Evaluated all three models on the test set using MSE and MAE
9. Plotted validation and training loss curves for all three models side-by-side for comparison

## Results Summary
The notebook prints MSE/MAE for:
- Baseline Model
- L2 Regularization Model
- Dropout Model

along with a side-by-side comparison of training and validation loss curves to visualize the effect of each regularization technique on overfitting.

## How to Run
```bash
pip install tensorflow scikit-learn matplotlib numpy
jupyter notebook DL_4.ipynb
```

## Author
Sukhada Bhoyar - 23070521033
