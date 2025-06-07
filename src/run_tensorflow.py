from hwinfo import collect_hw_info, get_cpu_time, get_memory_usage
import time
import tensorflow as tf

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


    cpu_time_before = get_cpu_time()
    memory_before = get_memory_usage()
    results = run_tensorflow_model(n_samples, n_features, batch_size, epochs, use_gpu)
    cpu_time_after = get_cpu_time()
    memory_after = get_memory_usage()

    return {
        "results": results,
        "cpu_time": cpu_time_after - cpu_time_before,
        "memory_usage": memory_after - memory_before,
    }
