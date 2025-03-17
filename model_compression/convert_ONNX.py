import torch
from torch import onnx
from torchvision import models
model = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
torch.save(model.state_dict(), "../models/VGG16.pt")

model = models.vgg16(num_classes=1000)
model.load_state_dict(torch.load("../models/VGG16.pt"))
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)
onnx.export(model = model, args = dummy_input, f="../models/VGG16.onnx")