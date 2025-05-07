import torch
from torchvision.models import mobilenet_v2

# Load model
model = mobilenet_v2()
model.classifier[1] = torch.nn.Linear(model.last_channel, 5)  # update class count if needed
model.load_state_dict(torch.load("mobilenetv2_maze.pth"))
model.eval()

# Dummy input
dummy_input = torch.randn(1, 3, 224, 224)

# Export to ONNX with opset 15
torch.onnx.export(
    model,
    dummy_input,
    "mobilenetv2_maze.onnx",
    export_params=True,
    opset_version=17,  # Required for IMX500
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output']
)

print("ONNX export complete with opset 17.")
