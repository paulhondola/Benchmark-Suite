import time
import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from utils import load_config, save_results
from hwinfo import collect_hw_info

def ml_train_infer_sklearn(n_samples=100_000, n_features=20):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    split = int(n_samples * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train = y[:split]

    model = LogisticRegression(max_iter=1000)
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()

    start_infer = time.perf_counter()
    _ = model.predict(X_test)
    end_infer = time.perf_counter()

    return {
        "train_time_sec": round(end_train - start_train, 4),
        "inference_time_sec": round(end_infer - start_infer, 4)
    }

def run_ml_benchmark():
    return ml_train_infer_sklearn()

def ml_train_infer_pytorch(batch_size=256, epochs=5, use_gpu=False):
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
    X = torch.randn(100_000, 20).to(device)
    y = torch.randint(0, 2, (100_000,)).to(device)
    X_train, X_test = X[:80_000], X[80_000:]
    y_train, y_test = y[:80_000], y[80_000:]

    model = nn.Sequential(
        nn.Linear(20, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_train = time.perf_counter()
    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
    end_train = time.perf_counter()

    start_infer = time.perf_counter()
    with torch.no_grad():
        _ = model(X_test)
    end_infer = time.perf_counter()

    return {
        "device": str(device),
        "train_time_sec": round(end_train - start_train, 4),
        "inference_time_sec": round(end_infer - start_infer, 4)
    }

def run_ml_benchmark_comparison():
    return {
        "scikit_learn": ml_train_infer_sklearn(),
        "pytorch_cpu": ml_train_infer_pytorch(use_gpu=False),
        "pytorch_gpu": ml_train_infer_pytorch(use_gpu=True)
    }

def run_all_ml_benchmarks():
    return {
        "ML Benchmark (scikit-learn)": run_ml_benchmark(),
        "ML Benchmark (sklearn vs PyTorch)": run_ml_benchmark_comparison()
    }

if __name__ == "__main__":
    results = run_all_ml_benchmarks()
    print("=== ML Benchmark Finished ===")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(RESULTS_DIR, f"ml_benchmarks_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved to: {out_path}")
