# DL_1 – Introduction to Deep Learning Frameworks

## Problem Statement
Explore the TensorFlow/Keras environment, create a basic neural network architecture, and understand the core building blocks of deep learning — tensors, layers, activation functions, and optimizers.

## Activities Performed
- Installed TensorFlow/Keras
- Understood tensors, layers, activation functions, and optimizers
- Built a simple regression model using `LinearRegression` from scikit-learn as an introductory exercise to model fitting and evaluation
- Understood model compilation and training concepts
- Evaluated the model using standard error metrics

## Tech Stack
- Python
- TensorFlow / Keras
- scikit-learn
- NumPy

## Workflow
1. Installed the `tensorflow` package
2. Created a simple dataset (`X`, `y`) for demonstration
3. Fit a `LinearRegression` model to the data
4. Extracted slope and intercept of the fitted line
5. Generated predictions and computed error metrics

## Results

**Equation of the fitted line:**
```
y = 14.5x + 7.5
```

**Error Metrics:**
| Metric | Value |
|--------|-------|
| MAE    | 1.40  |
| MSE    | 3.50  |
| RMSE   | 1.8708 |

## How to Run
```bash
pip install tensorflow scikit-learn numpy
jupyter notebook DL_1.ipynb
```

## Author
Sukhada Bhoyar - 23070521033
