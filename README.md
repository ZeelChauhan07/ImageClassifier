# Image-Classifier

# Fashion-MNIST Image Classifier

A simple feedforward neural network built with PyTorch to classify clothing items through image classification from the Fashion-MNIST dataset. This project was built as part of a 5-day ML hackathon to learn core deep learning concepts through hands-on implementation.

# Project Goal
Understand and implement the full ML pipeline — data loading, model design, training, and evaluation using a real dataset and framework, not just theory.

# Tech Stack
- Python 3.12
- PyTorch + torchvision
- Fashion-MNIST dataset

# Progress

- Loaded Fashion-MNIST dataset via `torchvision.datasets` and `DataLoader`
- Defined `SimpleNN`: a 3-layer feedforward neural network
  - Input: 784 (28x28 flattened image)
  - Hidden layers: 128 → 64
  - Output: 10 classes
- Implemented training loop (forward pass, loss calculation, backpropagation, optimizer step)
- Trained for 5 epochs — loss decreased from **0.5676 → 0.3099**
- Evaluated on unseen test set — **87.40% accuracy**

# Usage
- Run the training script (loads data, builds model, trains, evaluates)
  python ImageClassifier.py

# Project Structure

├── ImageClassifier.py            # Main script: data loading, modeltraining, evaluation
├── README.md            # Project documentation
└── data/                 # Fashion-MNIST dataset (auto-downloaded on first run)

# Model Architecture
SimpleNN(
(flatten): Flatten(start_dim=1, end_dim=-1)
(fc1): Linear(in_features=784, out_features=128)
(fc2): Linear(in_features=128, out_features=64)
(fc3): Linear(in_features=64, out_features=10)
)

# Next Steps
- Save and load trained model weights
- Single-image inference demo
- (Optional) Explore improvements: more epochs, CNN architecture, hyperparameter tuning

# How to Run
```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install torch torchvision
python ImageClassifier.py