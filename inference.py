import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.models.model import CNN
import os
# Set device and load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=device))
model.eval()
#Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])
#Prediction function
def predict_image(image_path):
    image = Image.open(image_path)

    image = transform(image)
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, 1)

    return prediction.item()
#Run example
if __name__ == "__main__":
    img_path = "data/samples/test.png"

    if os.path.exists(img_path):
        result = predict_image(img_path)
        print("Predicted Digit:", result)
    else:
        print("Image not found:", img_path)