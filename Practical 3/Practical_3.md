# DL_3 – Classification using Deep Feed Forward Network

## Problem Statement
Build a deep neural network classifier using the Breast Cancer dataset, and analyze its performance using a confusion matrix and accuracy score.

## Activities Performed
- Data preprocessing (train-test split, feature scaling)
- Designed and trained a DFFN classifier with Dropout layers
- Evaluated test accuracy and loss
- Analyzed the confusion matrix and classification accuracy

## Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- Seaborn, Matplotlib
- NumPy, Pandas

## Dataset
Breast Cancer Wisconsin dataset (`sklearn.datasets.load_breast_cancer`) — a binary classification task (malignant vs. benign).

## Workflow
1. Loaded the Breast Cancer dataset from scikit-learn
2. Split into train/test sets (80/20, stratified)
3. Standardized features with `StandardScaler`
4. Built a DFFN model:
   - `Dense(128, relu) → Dropout(0.3) → Dense(64, relu) → Dropout(0.3) → Dense(32, relu) → Dropout(0.2) → Dense(1, sigmoid)`
5. Compiled with Adam optimizer and binary cross-entropy loss
6. Trained for 50 epochs (batch size 32, 20% validation split)
7. Plotted training/validation accuracy and loss curves
8. Evaluated on the test set (loss & accuracy)
9. Generated predictions, computed the confusion matrix, and visualized it with a Seaborn heatmap
10. Verified accuracy using `accuracy_score`

## How to Run
```bash
pip install tensorflow scikit-learn seaborn matplotlib pandas numpy
jupyter notebook DL_3.ipynb
```

## Author
Sukhada Bhoyar - 23070521033
