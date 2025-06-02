import time
import platform
import torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import load_config, save_results
from hwinfo import collect_hw_info
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

def run_sklearn_model(model_name, n_samples, n_features, max_iter):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model_map = {
        "logistic_regression": LogisticRegression(max_iter=max_iter),
        "ridge": Ridge(max_iter=max_iter),
        "lasso": Lasso(max_iter=max_iter),
        "elasticnet": ElasticNet(max_iter=max_iter),
        "random_forest": RandomForestClassifier(n_estimators=100),
        "extra_trees": ExtraTreesClassifier(n_estimators=100),
        "gradient_boosting": GradientBoostingClassifier(),
        "svc": SVC(),
        "gaussian_nb": GaussianNB(),
        "knn": KNeighborsClassifier(),
        "mlp": MLPClassifier(max_iter=max_iter)
    }

    model = model_map.get(model_name)
    if model is None:
        return {"error": f"Unknown model type: {model_name}"}

    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    end_train = time.perf_counter()

    start_infer = time.perf_counter()
    y_pred = model.predict(X_test)
    end_infer = time.perf_counter()

    return {
        "Training Time (s)": round(end_train - start_train, 4),
        "Inference Time (s)": round(end_infer - start_infer, 4),
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
    }

def run_sklearn_workload(config):
    model_name = config["model"]
    n_samples = config["n_samples"]
    n_features = config["n_features"]
    max_iter = config["max_iter"]

    results = run_sklearn_model(model_name, n_samples, n_features, max_iter)

    return {
        "Config Metadata": config,
        "Benchmark Result": results,
        "System Info": collect_hw_info()
    }

def get_torch_device(use_gpu):
    if use_gpu:
        if platform.system() == "Darwin" and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")

def run_pytorch_model(n_samples, n_features, batch_size, epochs, use_gpu=False):
    device = get_torch_device(use_gpu)

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

    results = run_pytorch_model(n_samples, n_features, batch_size, epochs, use_gpu)

    return {
        "Config Metadata": config,
        "Benchmark Results": results,
        "System Info": collect_hw_info()
    }

def run_tensorflow_model(n_samples, n_features, batch_size, epochs, use_gpu):
    device_name = "/GPU:0" if use_gpu and tf.config.list_physical_devices("GPU") else "/CPU:0"

    x = tf.random.normal((n_samples, n_features))
    y = tf.random.uniform((n_samples,), maxval=2, dtype=tf.int32)
    y = tf.keras.utils.to_categorical(y, 2)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation="relu", input_shape=(n_features,)),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    start_train = time.perf_counter()
    with tf.device(device_name):
        model.fit(x, y, batch_size=batch_size, epochs=epochs, verbose=0)
    end_train = time.perf_counter()

    start_infer = time.perf_counter()
    with tf.device(device_name):
        _ = model.predict(x, verbose=0)
    end_infer = time.perf_counter()

    return {
        "Device": device_name,
        "Training Time (s)": round(end_train - start_train, 4),
        "Inference Time (s)": round(end_infer - start_infer, 4),
        "Accuracy": round(model.evaluate(x, y, verbose=0)[1], 4)
    }

def run_tensorflow_workload(config, use_gpu):
	n_samples = config["n_samples"]
	n_features = config["n_features"]
	batch_size = config["batch_size"]
	epochs = config["epochs"]

	results = run_tensorflow_model(n_samples, n_features, batch_size, epochs, use_gpu)

	return {
		"Config Metadata": config,
		"Benchmark Results": results,
		"System Info": collect_hw_info()
	}

def run_ml_workloads():
    config = load_config()
    config = config["ml_workloads"]

    sklearn = run_sklearn_workload(config["sklearn"])
    pytorch_cpu = run_pytorch_workload(config["pytorch"], use_gpu=False)
    pytorch_gpu = run_pytorch_workload(config["pytorch"], use_gpu=True)
    tf_cpu = run_tensorflow_workload(config["tensorflow"], use_gpu=False)
    tf_gpu = run_tensorflow_workload(config["tensorflow"], use_gpu=True)

    data = {
        "SKLearn": sklearn,
        "PyTorch CPU": pytorch_cpu,
        "PyTorch GPU": pytorch_gpu,
        "TensorFlow CPU": tf_cpu,
        "TensorFlow GPU": tf_gpu
    }

    save_results(data, "ml_workloads")

if __name__ == "__main__":
    print("Starting machine learning workloads")
    run_ml_workloads()
    print("Machine learning workloads finished")
