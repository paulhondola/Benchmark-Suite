from hwinfo import collect_hw_info, get_cpu_time, get_memory_usage
import time
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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

    cpu_time_before = get_cpu_time()
    memory_before = get_memory_usage()
    results = run_sklearn_model(model_name, n_samples, n_features, max_iter)
    cpu_time_after = get_cpu_time()
    memory_after = get_memory_usage()

    return {
        "results": results,
        "cpu_time": cpu_time_after - cpu_time_before,
        "memory_usage": memory_after - memory_before,
    }
