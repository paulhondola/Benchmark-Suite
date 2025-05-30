import torch
import time

def cnn_inference(batch_size=32, iterations=10):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(iterations):
            _ = model(x)
        end = time.perf_counter()
    return end - start
