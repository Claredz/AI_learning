from pathlib import Path

import numpy as np
import pandas as pd


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "gender_height_weight_100.csv"


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df[["height_cm", "weight_kg"]].to_numpy(dtype=float)
    y = df[["label_num"]].to_numpy(dtype=float)
    return X, y


def standardize_features(X):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def initialize_parameters(input_dim, hidden_dim, seed=42):
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0.0, 0.5, size=(input_dim, hidden_dim)),
        "b1": np.zeros((1, hidden_dim)),
        "W2": rng.normal(0.0, 0.5, size=(hidden_dim, 1)),
        "b2": np.zeros((1, 1)),
    }


def forward_pass(X, params):
    z1 = X @ params["W1"] + params["b1"]
    a1 = sigmoid(z1)
    z2 = a1 @ params["W2"] + params["b2"]
    y_pred = sigmoid(z2)

    cache = {
        "X": X,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "y_pred": y_pred,
    }
    return y_pred, cache


def compute_loss(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    loss = -(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred))
    return float(np.mean(loss))


def compute_accuracy(y_true, y_pred, threshold=0.5):
    predictions = (y_pred >= threshold).astype(int)
    return float(np.mean(predictions == y_true))


def backward_pass(y_true, cache, params):
    m = y_true.shape[0]
    y_pred = cache["y_pred"]
    a1 = cache["a1"]
    X = cache["X"]

    dz2 = y_pred - y_true
    dW2 = (a1.T @ dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    da1 = dz2 @ params["W2"].T
    dz1 = da1 * a1 * (1.0 - a1)
    dW1 = (X.T @ dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
    }


def update_parameters(params, grads, learning_rate):
    for key in params:
        params[key] -= learning_rate * grads[key]
    return params


def train(X, y, hidden_dim=4, learning_rate=0.1, epochs=5000, print_every=500):
    params = initialize_parameters(input_dim=X.shape[1], hidden_dim=hidden_dim)
    history = []

    for epoch in range(1, epochs + 1):
        y_pred, cache = forward_pass(X, params)
        loss = compute_loss(y, y_pred)
        accuracy = compute_accuracy(y, y_pred)
        grads = backward_pass(y, cache, params)
        params = update_parameters(params, grads, learning_rate)

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            print(f"epoch={epoch:4d} | loss={loss:.4f} | accuracy={accuracy:.4f}")

        history.append({"epoch": epoch, "loss": loss, "accuracy": accuracy})

    return params, history


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Data file not found: {DATA_PATH}\n"
            "Run `python experiments/mlp/generate_sample_data.py` first "
            "or place your CSV in data/raw/."
        )

    X, y = load_data(DATA_PATH)
    X_scaled, _, _ = standardize_features(X)
    _, history = train(X_scaled, y, hidden_dim=4, learning_rate=0.1, epochs=5000, print_every=500)

    final_metrics = history[-1]
    print("\nTraining finished.")
    print(f"final loss: {final_metrics['loss']:.4f}")
    print(f"final accuracy: {final_metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
