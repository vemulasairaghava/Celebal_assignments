from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

model = load_model("../models/mnist_autoencoder.keras")

print("Model Loaded Successfully!")

# denoised = model.predict(x_test_noisy)