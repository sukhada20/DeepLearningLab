# DL_2 – Housing Price Prediction using Deep Feed Forward Network

## Problem Statement
Predict house prices using a Deep Feed Forward Neural Network (DFFN) trained on the California housing dataset.

## Activities Performed
- Data preprocessing and normalization
- Designed multiple DFFN architectures of increasing depth
- Trained and evaluated each model using MAE, MSE, and RMSE
- Visualized training vs. validation loss curves

## Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- Pandas, NumPy
- Matplotlib

## Dataset
`california_housing_train.csv` (loaded from Colab's sample data), with `median_house_value` as the target variable.

## Workflow
1. Mounted Google Drive and loaded the housing dataset
2. Split features/target and performed an 80/20 train-test split
3. Standardized features using `StandardScaler`
4. Built and trained a primary DFFN model:
   - `128 → 64 → 32 → 1` (ReLU activations, Adam optimizer, MSE loss)
5. Evaluated using MAE, MSE, and RMSE
6. Plotted training vs. validation loss
7. Compared four different network architectures:

| Model | Architecture |
|-------|--------------|
| Model 1 | 64 → 1 |
| Model 2 | 128 → 64 → 1 |
| Model 3 | 128 → 64 → 32 → 1 |
| Model 4 | 256 → 128 → 64 → 32 → 1 |

Each model was trained for 100 epochs (batch size 32, 20% validation split) and evaluated with MAE, MSE, and RMSE, followed by individual loss plots.

## How to Run
```bash
pip install tensorflow scikit-learn pandas matplotlib
jupyter notebook DL_2.ipynb
```

> Note: The notebook was originally run in Google Colab and mounts Google Drive; adjust the data-loading path if running locally.

## Author
Sukhada Bhoyar - 23070521033
