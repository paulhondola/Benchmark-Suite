import torch
import time

def cnn_inference():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True).eval()
    x = torch.randn(32, 3, 224, 224)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(10):
            _ = model(x)
        end = time.perf_counter()
    return end - start
