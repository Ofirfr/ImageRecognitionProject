import os
import torch
import torchvision
from PIL import Image
from torchvision import transforms

# Define the transforms to be applied to the images
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize the image to (224, 224)
    transforms.ToTensor(),  # Convert the image to a tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
])

# Load the saved model
model = torchvision.models.resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Load and predict on all images in the test folder
test_folder = "testing/"
for filename in os.listdir(test_folder):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".webp"):
        img_path = os.path.join(test_folder, filename)
        img = Image.open(img_path).convert('RGB')
        img_tensor = transform(img)
        img_tensor = img_tensor.unsqueeze(0)  # Add a batch dimension
        output = model(img_tensor)
        _, predicted = torch.max(output.data, 1)
        class_names = ['donkey', 'horse', 'zebra']
        guess = class_names[predicted.item()]
        print(filename, guess, predicted.item())
