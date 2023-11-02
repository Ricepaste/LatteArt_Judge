import torch
from torchvision.io import read_image
import torchvision.models as models
from torchvision.models import EfficientNet_B1_Weights
# import torch

img = read_image(".\\main\\cropPhoto\\cropPhotocrop_2.jpg")

# Step 1: Initialize model with the best available weights
weights = EfficientNet_B1_Weights.DEFAULT
model = models.efficientnet_b1(weights=weights)

# experiment with the model
print(model._modules['classifier'])

# change the last layer of the model to fit our problem
model._modules['classifier'] = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(1280, 1))

print(model._modules['classifier'])

model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
prediction = model(batch).squeeze(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")


print(prediction.detach().numpy())
for i in range(prediction.detach().numpy().shape[0]):
    print(weights.meta["categories"][i])
