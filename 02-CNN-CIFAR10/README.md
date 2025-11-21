# CIFAR-10 Image Classification with CNN 🖼️

Implementation of a Convolutional Neural Network (CNN) to classify images into 10 categories (airplane, automobile, bird, etc.).

## Key Features
* **Data Augmentation:** Used `RandomFlip`, `RandomRotation`, and `RandomZoom` to reduce overfitting.
* **Optimization:** Implemented Learning Rate Scheduling to stabilize convergence.
* **Architecture:** Multiple Conv2D layers with Batch Normalization and Dropout.

## Results
* **Validation Accuracy:** ~81.08%
* **Validation Loss:** 0.6321 (stable)
* The model demonstrates solid generalization capabilities on the test set.