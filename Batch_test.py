import torch
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import os
import torch.nn.functional as F

# ---------- Settings ----------
MODEL_PATH = "mobilenetv2_maze.pth"
TEST_FOLDER = "test_images"  # Folder containing test images

# ---------- Define class names ----------
class_names = ['DE', 'FW', 'IN', 'LT', 'RT']

# ---------- Define image transformation ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---------- Load trained model ----------
model = mobilenet_v2(pretrained=False)
model.classifier[1] = torch.nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ---------- Load test images ----------
if not os.path.exists(TEST_FOLDER):
    raise FileNotFoundError(f"Test folder not found: {TEST_FOLDER}")

image_files = [f for f in os.listdir(TEST_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not image_files:
    print("❌ No test images found.")
else:
    print(f"🧪 Running predictions on {len(image_files)} images...\n")

# ---------- Predict each image ----------
for img_file in image_files:
    img_path = os.path.join(TEST_FOLDER, img_file)
    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

        predicted_class = class_names[predicted.item()]
        confidence_percent = confidence.item() * 100

        print(f"{img_file}: {predicted_class} ({confidence_percent:.2f}%)")
