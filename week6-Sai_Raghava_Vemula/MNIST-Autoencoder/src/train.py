from autoencoder import build_autoencoder

model = build_autoencoder()

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)