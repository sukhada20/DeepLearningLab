# DL_3 – MNIST Classification using Deep Feed Forward Network

## Problem Statement
Build a Deep Feed Forward Neural Network (DFFN) for handwritten digit classification using the MNIST dataset, and analyze the effect of different hyperparameters such as learning rate, optimizer, batch size, activation function, number of neurons, and number of epochs on model performance.

## Activities Performed
* Loaded and preprocessed the MNIST handwritten digit dataset
* Normalized image pixel values to the range `[0, 1]`
* Designed and trained multiple DFFN classifiers with different hyperparameter configurations
* Experimented with Adam and SGD optimizers
* Compared ReLU and Tanh activation functions
* Evaluated training, validation, and test accuracy
* Compared test loss across different model configurations
* Analyzed the effect of batch size, learning rate, network architecture, activation function, and training epochs

## Tech Stack
* Python
* TensorFlow / Keras
* NumPy
* Jupyter Notebook / Google Colab

## Dataset
**MNIST Handwritten Digit Dataset** — a multiclass image classification dataset containing grayscale images of handwritten digits from `0` to `9`.
* Image size: `28 × 28` pixels
* Number of classes: `10`
* Input features after flattening: `784`
* Output layer: `10` neurons with Softmax activation
The dataset was loaded directly using:
```python
tf.keras.datasets.mnist.load_data()
```

## Workflow
1. Loaded the MNIST dataset using TensorFlow/Keras.
2. Separated the data into training and testing sets.
3. Normalized pixel values by dividing them by `255.0`.
4. Flattened each `28 × 28` image into a `784`-element vector using the `Flatten` layer.
5. Built a Deep Feed Forward Neural Network with four hidden layers.
6. Used ReLU or Tanh activation functions in the hidden layers.
7. Used a `Dense(10, softmax)` output layer for multiclass digit classification.
8. Compiled the models using either Adam or SGD optimizers with `sparse_categorical_crossentropy` loss.
9. Trained each model using an 80/20 training-validation split.
10. Evaluated each trained model on the MNIST test dataset.
11. Recorded training accuracy, validation accuracy, test accuracy, and test loss.
12. Compared the performance of all four experimental configurations.


## How to Run
```bash
pip install tensorflow numpy
jupyter notebook DL_3.ipynb
```
Alternatively, the notebook can be executed directly in **Google Colab**.

## Author
Sukhada Bhoyar - 23070521033
