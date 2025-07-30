# MNIST Digit Classifier

A neural network-based digit classifier built with TensorFlow/Keras to recognize handwritten digits (0-9) from the MNIST dataset.

## Overview

This project implements a deep learning model that can classify handwritten digits with high accuracy. The model is trained on the famous MNIST dataset, which contains 70,000 images of handwritten digits (60,000 for training and 10,000 for testing).

## Features

- **Deep Neural Network**: Multi-layer perceptron with ReLU activation
- **High Accuracy**: Achieves ~97%+ accuracy on test data
- **Data Visualization**: Includes matplotlib visualizations of digits and training metrics
- **Performance Monitoring**: Tracks loss and accuracy during training

## Model Architecture

The neural network consists of:
- **Input Layer**: Flattened 28×28 pixel images (784 features)
- **Hidden Layer 1**: 128 neurons with ReLU activation
- **Hidden Layer 2**: 32 neurons with ReLU activation  
- **Output Layer**: 10 neurons with softmax activation (one for each digit class)

## Dataset

- **Source**: MNIST handwritten digit database
- **Training Set**: 60,000 images
- **Test Set**: 10,000 images
- **Image Size**: 28×28 pixels, grayscale
- **Classes**: 10 (digits 0-9)

## Requirements

```
tensorflow>=2.19.0
matplotlib>=3.10.0
scikit-learn>=1.7.1
numpy>=2.1.0
jupyter
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/ayeshowcode/digit-classifier.git
cd digit-classifier
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
```

3. Install required packages:
```bash
pip install tensorflow matplotlib scikit-learn numpy jupyter
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook main.ipynb
```

2. Run all cells to:
   - Load and preprocess the MNIST dataset
   - Build and train the neural network
   - Evaluate model performance
   - Visualize results and predictions

## Key Code Components

### Data Preprocessing
- Loads MNIST dataset using `keras.datasets.mnist.load_data()`
- Normalizes pixel values to [0, 1] range by dividing by 255
- No additional preprocessing required (data comes pre-split)

### Model Training
- Uses sparse categorical crossentropy loss function
- Adam optimizer for efficient gradient descent
- Trains for 25 epochs with 20% validation split
- Monitors both accuracy and loss metrics

### Evaluation
- Tests on 10,000 unseen images
- Calculates overall accuracy using scikit-learn
- Provides confusion analysis for misclassified digits

## Results

The model typically achieves:
- **Training Accuracy**: ~98%+
- **Validation Accuracy**: ~97%+
- **Test Accuracy**: ~97%+

## Visualization Features

The notebook includes visualizations for:
- Sample digit images from the dataset
- Training/validation loss curves
- Training/validation accuracy curves
- Individual prediction examples
- Misclassified digit analysis

## Project Structure

```
digit-classifier/
│
├── main.ipynb          # Main Jupyter notebook with complete implementation
├── README.md           # Project documentation (this file)
└── .venv/              # Virtual environment (if created)
```

## Potential Improvements

- **Convolutional Neural Network**: Could improve accuracy by preserving spatial information
- **Data Augmentation**: Rotate, shift, or scale images to increase dataset diversity
- **Regularization**: Add dropout layers to prevent overfitting
- **Hyperparameter Tuning**: Optimize learning rate, batch size, and architecture
- **Model Persistence**: Save trained models for future use

## Technical Notes

- The model flattens 2D images to 1D vectors, which loses spatial information
- Uses dense (fully connected) layers throughout
- Softmax activation in output layer provides probability distribution over classes
- Model is suitable for educational purposes and understanding basic neural networks

## License

This project is open source and available under the [MIT License](LICENSE).

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ayeshowcode/digit-classifier/issues).

## Author

**Ayesh** - [GitHub Profile](https://github.com/ayeshowcode)

---

*This project was created as part of a Machine Learning course assignment.*
