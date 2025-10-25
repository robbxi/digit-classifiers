# Digit Classifiers

A comprehensive implementation and comparison of various machine learning classifiers for handwritten digit recognition using the MNIST dataset.

## Overview

This project implements and compares five different classification approaches for recognizing handwritten digits (0-9):

1. **K-Nearest Neighbors (KNN)** - Instance-based learning using distance metrics
2. **Naive Bayes** - Probabilistic classifier using Bayes' theorem
3. **Linear Classifier** - Single-layer neural network
4. **Multi-layer Perceptron** - Deep neural network with hidden layers
5. **Convolutional Neural Network (CNN)** - Deep learning with convolutional layers

## Features

- **Interactive Demo**: Draw digits and get real-time predictions from any classifier
- **Performance Testing**: Evaluate accuracy on test datasets with customizable sample sizes
- **Confusion Matrix**: Visualize classification errors and patterns
- **Model Comparison**: Test all models simultaneously and compare results

## Quick Start

### Using Google Colab (Recommended)

1. Open the notebook in Google Colab: 1. [Open the notebook in Google Colab](https://colab.research.google.com/github/robbxi/digit-classifiers/blob/main/notebook/Project_1.ipynb)
2. Run the installation cell to set up dependencies
3. Run the Interactive Demo cell
4. Draw a digit in the canvas and click "Predict"

### Local Setup
```bash
# Clone the repository
git clone https://github.com/robbxi/digit-classifiers.git
cd digit-classifiers/notebook

# Install dependencies
pip install numpy torch torchvision pillow ipycanvas ipywidgets

# Run Jupyter notebook
jupyter notebook
```

## Project Structure
```
digit-classifiers/
├── notebook/
│   ├── knn.py                    # K-Nearest Neighbors implementation
│   ├── naive_bayes.py            # Naive Bayes classifier
│   ├── linear_classifier.py      # Linear classifier with PyTorch
│   ├── ml_perceptron.py          # Multi-layer perceptron
│   ├── cnn.py                    # Convolutional Neural Network
│   ├── utils.py                  # Data loading and utilities
│   └── MNIST/                    # MNIST dataset (auto-downloaded)
└── README.md
```

## Using the Interactive Demo

### Drawing Canvas

1. **Draw**: Click and drag on the black canvas to draw a digit
2. **Select Model**: Choose a classifier from the dropdown menu
3. **Predict**: Click the "Predict" button to classify your drawing
4. **Clear**: Clear the canvas to draw a new digit
5. **All Models**: Select "All" to see predictions from all classifiers

### Testing Accuracy

1. **Select Model**: Choose which classifier to test (or "All" for comparison)
2. **Set Sample Size**: Use the slider to select number of test samples (50-5000)
3. **Confusion Matrix**: Check the box to see detailed classification errors
4. **Test**: Click "Test Accuracy" to evaluate performance

## Model Details

### K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Key Parameter**: k=3 neighbors
- **Pros**: Simple, no training required
- **Cons**: Slow prediction, memory intensive

### Naive Bayes
- **Type**: Probabilistic classifier
- **Approach**: Binary threshold (pixel on/off)
- **Pros**: Fast training, interpretable
- **Cons**: Slower inference, assumes independence

### Linear Classifier
- **Type**: Single-layer neural network
- **Architecture**: 784 → 10 neurons
- **Activation**: None (logits)
- **Training**: SGD optimizer, MSE loss

### Multi-layer Perceptron
- **Type**: Deep neural network
- **Architecture**: 784 → hidden layers → 10
- **Activation**: ReLU
- **Training**: Backpropagation with dropout

### Convolutional Neural Network (CNN)
- **Type**: Deep learning with spatial features
- **Architecture**: Conv layers → Pooling → FC layers
- **Activation**: ReLU
- **Training**: Adam optimizer, cross-entropy loss
- **Performance**: Highest accuracy

## Results

Typical accuracy on MNIST test set:

| Model | Accuracy | Speed |
|-------|----------|-------|
| KNN | ~95% | Slow |
| Naive Bayes | ~84% | Very Slow |
| Linear | ~92% | Fast |
| Multi-layer | ~97% | Fast |
| CNN | ~99% | Very Fast |

*Note: CNN is fastest at inference despite being the most complex, due to GPU optimization*

## Dataset

This project uses the **MNIST dataset**:
- 60,000 training images
- 10,000 test images
- 28x28 grayscale images
- 10 classes (digits 0-9)

The dataset is automatically downloaded on first run.

## Requirements
```
numpy>=1.19.0
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
ipycanvas>=0.13.0 (for interactive demo)
ipywidgets>=7.6.0 (for interactive demo)
```

## Troubleshooting

### Canvas Not Working
If the interactive canvas doesn't work on first run:
1. Make sure you've run the cell to enable custom widget manager
2. Try refreshing the page and running all cells again
3. Use the HTML Canvas version (included in notebook)

### CUDA Errors
If you see CUDA device errors when loading models:
- Models are saved with GPU weights but Colab may run on CPU
- The code automatically handles this with `map_location=torch.device('cpu')`

### Import Errors
Make sure you're in the correct directory:
```bash
cd /content/digit-classifiers/notebook
```

## Contributing

Contributions are welcome! Areas for improvement:
- Additional classifiers (SVM, Random Forest, etc.)
- Hyperparameter optimization
- Data augmentation
- Model ensembling
- Performance optimizations

## License

This project is open source and available under the MIT License.


## Contact

For questions or issues, please open an issue on GitHub.

---

**Note**: This is an educational project demonstrating various machine learning approaches. For production digit recognition, use established libraries like scikit-learn or pre-trained models.
