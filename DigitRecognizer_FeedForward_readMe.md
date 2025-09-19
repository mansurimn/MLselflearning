# 🧠 Simple Neural Network on MNIST (TensorFlow)

This project implements a **basic feedforward neural network** using TensorFlow/Keras to classify handwritten digits from the **MNIST dataset**.

---

## 📂 Project Overview
- **Dataset**: MNIST (28x28 grayscale images of digits 0–9).
- **Goal**: Train a simple NN to recognize digits with high accuracy.
- **Framework**: TensorFlow / Keras.

---

## 🏗 Model Architecture
- Input Layer: Flatten (28x28 → 784)
- Dense Layer 1: 128 neurons, ReLU activation
- Dense Layer 2: 64 neurons, ReLU activation
- Output Layer: 10 neurons, Softmax activation

---

## ⚙️ Training
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Metrics: Accuracy
- Epochs: 3–5

---

## 📊 Results
- **Training Accuracy**: ~98%
- **Test Accuracy**: ~97%

---

## 🚀 Usage
```python
# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate
model.evaluate(x_test, y_test)

# Predict
predictions = model.predict(x_test[:5])


We’re training a Feedforward Neural Network to recognize handwritten digits (0–9) from the MNIST dataset.

1. Dataset (MNIST)

Contains 70,000 grayscale images of handwritten digits (28×28 pixels).

Each image has a label (0–9).

Example: an image of “7” has label 7.

2. Preprocessing

Normalize pixel values from 0–255 → 0–1 (makes training faster & stable).

Flatten images: 28×28 → 784 numbers (input to NN).

3. Neural Network Architecture

Input Layer: 784 neurons (one for each pixel).

Hidden Layers:

Dense(128, ReLU)

Dense(64, ReLU)

Output Layer: Dense(10, Softmax) → gives probabilities for each digit (0–9).

4. Training Process

Loss Function: SparseCategoricalCrossentropy (good for multi-class problems).

Optimizer: Adam (smart gradient descent).

Training: Feed images in small batches, update weights using backpropagation + gradient descent.

After ~3–5 epochs → model learns to recognize digits with 97–98% accuracy 🎯.

5. Evaluation

Test on unseen data (10,000 images).

Print accuracy → ~97%.

6. Predictions

Feed new images → model outputs probabilities.

Pick the class with highest probability → predicted digit.

We also plotted the first 5 test images with predicted vs. true labels.
