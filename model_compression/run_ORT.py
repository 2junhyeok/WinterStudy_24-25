import time
import torch
import onnxruntime as ort
from PIL import Image
from torchvision import models
from torchvision import transforms


def to_numpy(tensor):
    return tensor.detach().cpu().numpy()


image = Image.open("../datasets/images/cat.jpg")
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48235, 0.45882, 0.40784],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)
input = transform(image).unsqueeze(0)

model = models.vgg16(num_classes=1000)
model.load_state_dict(torch.load("../models/VGG16.pt"))
model.eval()

with torch.no_grad():
    start_time = time.time()
    output = model(input)
    end_time = time.time()
    print("파이토치:")
    print(output.max())
    print(end_time - start_time)

ort_session = ort.InferenceSession("../models/VGG16.onnx")
start_time = time.time()
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(input)}
ort_outs = ort_session.run(output_names=None, input_feed=ort_inputs)
end_time = time.time()
print("ONNX:")
print(max(ort_outs[0][0]))
print(end_time - start_time)
