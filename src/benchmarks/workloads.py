import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

def sort_numpy_array(size=10_000_000):
    arr = np.random.rand(size)
    start = time.perf_counter()
    sorted_arr = np.sort(arr)
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def sort_pandas_dataframe(rows=1_000_000):
    df = pd.DataFrame({
        "id": np.arange(rows),
        "value": np.random.rand(rows)
    })
    start = time.perf_counter()
    df_sorted = df.sort_values(by="value")
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def compare_sorting_methods():
    return {
        "numpy_sort": sort_numpy_array(),
        "pandas_sort": sort_pandas_dataframe()
    }

def join_pandas_dataframes(left_size=1_000_000, right_size=500_000):
    left = pd.DataFrame({
        "id": np.arange(left_size),
        "value_left": np.random.rand(left_size)
    })

    right = pd.DataFrame({
        "id": np.random.choice(np.arange(left_size), size=right_size, replace=False),
        "value_right": np.random.rand(right_size)
    })

    start = time.perf_counter()
    joined = pd.merge(left, right, on="id", how="inner")
    end = time.perf_counter()
    return {"execution_time_sec": round(end - start, 4)}

def run_join_benchmark():
    return {
        "pandas_inner_join": join_pandas_dataframes()
    }

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
