# 🎓 Deep Learning Teaching Tool

> GradientMonitoring: Real-Time Neural Network Learning Platform
> CS7050: Data Warehousing & Mining - Fall 2025  
> Kennesaw State University

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red.svg)](https://streamlit.io/)
[![NumPy](https://img.shields.io/badge/NumPy-1.24-orange.svg)](https://numpy.org/)

## 🎯 Project Overview

Deep Learning Teaching Tool is a web-based educational application built with Python and Streamlit that visualizes how neural networks learn through backpropagation. It provides real-time visualization of gradient descent optimization, weight updates, and decision boundary evolution on 2D classification datasets.

## ✨ Features

- **Real-time Training Visualization**: Watch neural networks learn with each epoch
- **Interactive Decision Boundaries**: See how predictions evolve during training
- **Multiple Datasets**: XOR, Circles, and Moons datasets
- **Adjustable Hyperparameters**: Learning rate, network size, training epochs
- **Loss Tracking**: Real-time loss curve visualization
- **Network Architecture Visualization**: See weights and activations
- **Algorithm Pseudo-code**: Backpropagation steps displayed

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone or download the project
cd deep-learning-teaching-tool

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

## 📁 Project Structure

```
deep-learning-teaching-tool/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── utils/
│   ├── __init__.py
│   ├── neural_network.py      # Neural network implementation
│   └── datasets.py            # Dataset generators
└── components/
    ├── __init__.py
    └── visualization.py       # Plotting functions
```

## 📚 File Descriptions

### Core Files

#### `app.py`
Main Streamlit application that orchestrates the UI and training loop.

#### `utils/neural_network.py`
Contains the `NeuralNetwork` class implementing:
- Forward propagation
- Backpropagation
- Weight updates
- Loss computation
- Prediction methods

#### `utils/datasets.py`
Dataset generation functions:
- `generate_xor()`: 4-point XOR dataset
- `generate_circles()`: Concentric circles (100 points)
- `generate_moons()`: Interleaving moons (100 points)

#### `components/visualization.py`
Visualization functions using Matplotlib:
- `plot_decision_boundary()`: Decision boundary with data points
- `plot_loss_curve()`: Training loss over epochs
- `plot_network_architecture()`: Network structure visualization

## 🎮 How to Use

1. **Launch Application**: Run `streamlit run app.py`
2. **Select Dataset**: Choose from XOR, Circles, or Moons in sidebar
3. **Configure Network**: Adjust hidden layer size (2-12 neurons)
4. **Set Hyperparameters**: 
   - Learning rate (0.01 - 1.0)
   - Epochs per training step (1-100)
5. **Train Network**: Click "Train Network" button
6. **Observe Results**: Watch decision boundary evolve and loss decrease
7. **Reset**: Click "Reset Network" to start fresh

## 🧮 Datasets

### XOR (4 points)
Classic non-linearly separable problem demonstrating the necessity of hidden layers.
- Points: {(0.2,0.2)→0, (0.2,0.8)→1, (0.8,0.2)→1, (0.8,0.8)→0}

### Circles (100 points)
Concentric circles testing radial decision boundaries.
- Inner circle (r=0.25): Class 0
- Outer circle (r=0.65): Class 1

### Moons (100 points)
Interleaving crescents challenging complex curve learning.
- Upper moon: Class 0
- Lower moon: Class 1

## 🔬 Algorithm

The tool implements backpropagation with stochastic gradient descent:

1. **Forward Pass**: Compute activations layer-by-layer using sigmoid activation
2. **Loss Computation**: Binary cross-entropy loss
3. **Backward Pass**: Compute gradients using chain rule
4. **Weight Update**: Gradient descent with configurable learning rate

### Mathematical Formulation

**Forward Pass:**
```
h = σ(W₁·x + b₁)
ŷ = σ(W₂·h + b₂)
```

**Loss:**
```
L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

**Backpropagation:**
```
δ₂ = (ŷ - y)·σ'(ŷ)
δ₁ = (W₂ᵀ·δ₂)·σ'(h)
```

**Weight Update:**
```
W₂ = W₂ - α·∇W₂L
W₁ = W₁ - α·∇W₁L
```

## 🛠️ Technologies

- **Streamlit**: Web application framework
- **NumPy**: Numerical computation
- **Matplotlib**: Data visualization
- **Python**: Core programming language

## 📊 Network Architecture

- **Input Layer**: 2 neurons (x, y coordinates)
- **Hidden Layer**: 2-12 neurons (configurable)
- **Output Layer**: 1 neuron (binary classification)
- **Activation**: Sigmoid function σ(x) = 1/(1+e^(-x))
- **Loss**: Binary cross-entropy

## 🎓 Educational Value

This tool helps students understand:
- How backpropagation propagates gradients backward through the network
- Why hidden layers enable learning of non-linear decision boundaries
- The effect of learning rate on convergence speed and stability
- Real-time evolution of decision boundaries during training
- Relationship between loss minimization and model performance

## 📦 Deployment

### Deploy to Streamlit Cloud (Free)

1. Push code to GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository
5. Set main file: `app.py`
6. Click "Deploy"

### Local Deployment

```bash
# Run on custom port
streamlit run app.py --server.port 8080

# Run with different configuration
streamlit run app.py --server.headless true
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Module not found error
```bash
# Solution: Install dependencies
pip install -r requirements.txt
```

**Issue**: Streamlit command not found
```bash
# Solution: Ensure streamlit is installed
pip install streamlit
# Or use python -m
python -m streamlit run app.py
```

**Issue**: Port already in use
```bash
# Solution: Use different port
streamlit run app.py --server.port 8502
```

## 💡 Training Tips

- **Higher learning rate** (0.7-1.0): Faster training but may overshoot
- **Lower learning rate** (0.01-0.3): Slower but more stable convergence
- **More hidden neurons**: Increases model capacity but may overfit
- **XOR dataset**: Requires minimum 2 hidden neurons to solve
- **Reset network**: If stuck in poor local minimum

## 📝 License

This project is for educational purposes as part of CS7050 coursework.

## 👨‍💻 Author

**Kokou Adje**  
CS7050 - Fall 2025  
Kennesaw State University  
Email: kadje@students.kennesaw.edu

## 🙏 Acknowledgments

- Course: CS7050 Data Warehousing & Mining
- Institution: Kennesaw State University
- Inspired by: TensorFlow Playground, Neural Network Playground

---

**Project Name**: Deep Learning Teaching Tool  
**Repository**: https://github.com/Kokou-Adje/deep-learning-teaching-tool
**Last Updated**: December 2025