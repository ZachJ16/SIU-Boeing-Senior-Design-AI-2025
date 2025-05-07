import time
import torch
import cv2
from torchvision import transforms
from torch.autograd import Variable
import os
import json

# Load the trained model (MobileNetV2)
model = torch.load("mobilenetv2_maze.pth")
model.eval()  # Set model to evaluation mode

# Define class labels (from your previous network classes)
class_labels = ['DE', 'FW', 'IN', 'LT', 'RT']

# Create a directory to store the captured images and predictions
if not os.path.exists('captured_data'):
    os.makedirs('captured_data')

# Initialize the camera (Use libcamera or OpenCV with V4L2)
# Assuming you are using OpenCV with the V4L2 interface on Raspberry Pi
camera = cv2.VideoCapture("/dev/video0")  # IMX500 camera interface (may vary, check your system)

# Check if the camera is opened correctly
if not camera.isOpened():
    print("Error: Could not open camera.")
    exit()

# Image preprocessing for MobileNetV2
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

# Get neural network's decision and confidence
def neural_network_decision(image):
    # Preprocess the image for the neural network
    image_tensor = preprocess_image(image)

    # Convert to Variable to send through the model
    image_tensor = Variable(image_tensor)

    # Get the model's predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1).max().item()  # Get confidence value
    
    return predicted_class.item(), confidence

# Main loop to capture images and make predictions
while True:
    # Capture a frame from the camera
    ret, frame = camera.read()

    if not ret:
        print("Failed to capture image")
        break

    # Get the neural network's decision and confidence value
    decision, confidence = neural_network_decision(frame)
    label = class_labels[decision]  # Get the corresponding label for the decision

    # Save image with timestamp and prediction details
    timestamp = time.time()
    image_filename = f"captured_data/image_{timestamp}.jpg"
    json_filename = f"captured_data/prediction_{timestamp}.json"
    
    # Save the image
    cv2.imwrite(image_filename, frame)

    # Save the prediction and confidence in a JSON file
    prediction_data = {
        'timestamp': timestamp,
        'prediction': label,
        'confidence': confidence
    }
    
    with open(json_filename, 'w') as f:
        json.dump(prediction_data, f, indent=4)

    # Print the result
    print(f"Captured image: {image_filename}")
    print(f"Prediction: {label} | Confidence: {confidence:.4f}")

    # Delay between captures (optional)
    time.sleep(1)  # Adjust the delay as needed

# Release the camera when done
camera.release()
cv2.destroyAllWindows()
