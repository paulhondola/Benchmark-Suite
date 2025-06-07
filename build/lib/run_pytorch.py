import time
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from hwinfo import collect_hw_info, get_cpu_time, get_memory_usage

def get_pytorch_device(use_gpu):
    if use_gpu:
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")

def run_pytorch_model(n_samples, n_features, batch_size, epochs, use_gpu=False):
    device = get_pytorch_device(use_gpu)

    if device.type == 'mps':
        print("Using Apple Silicon GPU (MPS) for PyTorch.")
    elif device.type == 'cuda':
        print("Using NVIDIA GPU for PyTorch.")
    else:
        print("Using CPU for PyTorch.")

    X = torch.randn(n_samples, n_features).to(device)
    y = torch.randint(0, 2, (n_samples,)).to(device)
    split = int(0.8 * n_samples)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = nn.Sequential(
        nn.Linear(n_features, 64),
        nn.ReLU(),
        nn.Linear(64, 2)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_train = time.perf_counter()
    for _ in range(epochs):
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
        outputs = model(X_test)
        y_pred = torch.argmax(outputs, dim=1)
    end_infer = time.perf_counter()

    correct = (y_pred == y_test).sum().item()
    accuracy = correct / y_test.size(0)

    return {
        "Device": str(device),
        "Training Time (s)": round(end_train - start_train, 4),
        "Inference Time (s)": round(end_infer - start_infer, 4),
        "Accuracy": round(accuracy, 4)
    }

def run_pytorch_workload(config, use_gpu):
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    batch_size = config["batch_size"]
    epochs = config["epochs"]

    cpu_time_before = get_cpu_time()
    memory_before = get_memory_usage()
    results = run_pytorch_model(n_samples, n_features, batch_size, epochs, use_gpu)
    cpu_time_after = get_cpu_time()
    memory_after = get_memory_usage()

    return {
        "results": results,
        "cpu_time": cpu_time_after - cpu_time_before,
        "memory_usage": memory_after - memory_before,
    }
