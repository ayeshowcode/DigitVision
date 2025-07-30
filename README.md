# MNIST Digit Classifier

A neural network that classifies handwritten digits (0-9) using TensorFlow/Keras on the MNIST dataset.

## Features
- Achieves ~97% accuracy on test data
- Simple 3-layer neural network (128→32→10 neurons)
- Includes data visualization and training metrics

## Quick Start

1. Install dependencies:
```bash
pip install tensorflow matplotlib scikit-learn numpy jupyter
```

2. Run the notebook:
```bash
jupyter notebook main.ipynb
```

## Model Architecture
- **Input**: 28×28 pixel images (flattened to 784 features)
- **Hidden Layers**: 128 neurons → 32 neurons (ReLU activation)
- **Output**: 10 neurons (softmax for digit classification)

## Results
- Training accuracy: ~98%
- Test accuracy: ~97%
- Dataset: 60,000 training + 10,000 test images

---
