import os
from flask import Flask, render_template, request
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from flask import jsonify

app = Flask(__name__)

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file from the form
    uploaded_file = request.files['file']
    # Save the uploaded file to a temporary directory
    filename = uploaded_file.filename
    file_path = os.path.join('static', 'uploads', filename)
    uploaded_file.save(file_path)
    
    # Open the uploaded image and apply the transform
    image = Image.open(file_path)
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    
    # Make a prediction using the model
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    prediction = predicted.item()
    
    # Map the prediction index to a label
    if prediction == 0:
        label = 'donkey'
    elif prediction == 1:
        label = 'horse'
    else:
        label = 'zebra'

    return jsonify(result=label)

if __name__ == '__main__':
    app.run(debug=True)
