# 🧠 Autoencoder for Image Denoising using MNIST

## 📌 Project Overview

This project implements a **Convolutional Autoencoder** using TensorFlow/Keras to remove Gaussian noise from handwritten digit images in the MNIST dataset.

The model is trained to reconstruct clean images from noisy inputs by learning compressed latent representations and decoding them back into denoised images.

---

## 🎯 Objective

- Load the MNIST PNG dataset.
- Add Gaussian noise to the images.
- Build a Convolutional Autoencoder.
- Train the model to reconstruct clean images.
- Evaluate the denoising performance.
- Save the trained model for future inference.

---

## 📂 Dataset

**Dataset:** MNIST Handwritten Digits

- Training Images: 60,000
- Testing Images: 10,000
- Image Size: 28 × 28 pixels
- Color Format: Grayscale

Dataset Structure:

```
mnist_png/
│
├── training/
│   ├── 0
│   ├── 1
│   ├── ...
│   └── 9
│
└── testing/
    ├── 0
    ├── 1
    ├── ...
    └── 9
```

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Kaggle Notebook (GPU)

---

## 🧠 Model Architecture

### Encoder

- Conv2D (32 filters, 3×3, ReLU)
- MaxPooling2D
- Conv2D (32 filters, 3×3, ReLU)
- MaxPooling2D

### Decoder

- Conv2D (32 filters, 3×3, ReLU)
- UpSampling2D
- Conv2D (32 filters, 3×3, ReLU)
- UpSampling2D
- Conv2D (1 filter, 3×3, Sigmoid)

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|--------|
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Epochs | 20 |
| Batch Size | 128 |
| Noise Type | Gaussian Noise |
| Noise Factor | 0.5 |

---

## 📊 Training Results

- Training completed successfully for 20 epochs.
- Final Training Loss: **0.0952**
- Final Validation Loss: **0.0946**

The validation loss consistently decreased during training, indicating that the model learned to reconstruct clean images effectively.

---

## 📁 Project Structure

```
MNIST-Autoencoder/
│
├── dataset/
│   └── mnist_png/
│
├── models/
│   └── mnist_autoencoder.keras
│
├── outputs/
│   ├── comparison.png
│   ├── denoised.png
│   ├── noisy.png
│   ├── original.png
│   ├── original_vs_noisy.png
│   └── loss_curve.png
│
├── Autoencoder_Image_Denoising.ipynb
├── requirements.txt
└── README.md
```

---

## 📈 Workflow

```
Load MNIST Dataset
        │
        ▼
Normalize Images
        │
        ▼
Add Gaussian Noise
        │
        ▼
Build Autoencoder
        │
        ▼
Train Model
        │
        ▼
Generate Denoised Images
        │
        ▼
Save Model & Results
```

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/vemulasairaghava/MNIST-Autoencoder.git
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Run:

```
Autoencoder_Image_Denoising.ipynb
```

using Jupyter Notebook, VS Code, or upload it to Kaggle.

---

## 📷 Results

The model successfully reconstructs clean handwritten digits from noisy images.

Example outputs include:

- Original Image
- Noisy Image
- Denoised Image
- Training Loss Curve

Images are available in the **outputs/** folder.

---

## 💾 Saved Model

The trained model is saved as:

```
models/mnist_autoencoder.keras
```

This model can be loaded later for image denoising without retraining.

---

## 📚 Learning Outcomes

Through this project, I learned:

- Image preprocessing
- Gaussian noise generation
- Convolutional Autoencoders
- CNN Encoder–Decoder architecture
- Image reconstruction
- Model training using TensorFlow/Keras
- Saving and reusing trained deep learning models

---

## 👨‍💻 Author

**Sai Raghava Vemula**

GitHub: https://github.com/vemulasairaghava