# 🧠 MNIST Autoencoder

An **autoencoder** is a neural network that learns to **compress data (encode)** into a smaller representation and then **reconstruct (decode)** it back.  
In this project, we train an autoencoder on the **MNIST handwritten digits dataset** to reconstruct digits from their compressed form.

---

## 📌 Project Overview
- **Goal**: Train a neural network to reconstruct handwritten digits (unsupervised learning).
- **Dataset**: MNIST (70,000 grayscale images of digits 0–9).
- **Architecture**:
  - **Encoder**: Reduces 784 pixel values → 64 compressed values.
  - **Decoder**: Expands 64 compressed values → 784 pixel values.
- **Loss**: Binary Crossentropy  
- **Optimizer**: Adam

---

## ⚙️ Installation
```bash
# Clone repo
git clone https://github.com/yourusername/MNIST-Autoencoder.git
cd MNIST-Autoencoder

# Install dependencies
pip install tensorflow matplotlib numpy

Code Outline

Load dataset

from tensorflow.keras.datasets import mnist
(x_train, _), (x_test, _) = mnist.load_data()


Normalize to [0,1], flatten 28×28 → 784.

Build Autoencoder

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

input_layer = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')


Train

autoencoder.fit(
    x_train, x_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)


Visualize Results

reconstructed = autoencoder.predict(x_test)
# Plot original vs reconstructed

📊 Results

Top row → Original digits

Bottom row → Reconstructed digits

Even though reconstructions may look slightly blurry, the digits are recognizable.

🌍 Applications of Autoencoders

🔹 Noise removal (denoising images)

🔹 Anomaly detection (fraud, defective items)

🔹 Compression (store images efficiently)

🔹 Feature extraction for downstream ML tasks

📚 References

MNIST Dataset

TensorFlow Autoencoder Tutorial

Goodfellow et al., Deep Learning
