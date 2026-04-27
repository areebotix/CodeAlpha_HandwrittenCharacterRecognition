import torch
from src.models.model import CNN

model = CNN()
model.load_state_dict(torch.load("models/mnist_cnn.pth"))
model.eval()

def predict(image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        return torch.argmax(output, 1).item()
            