from torchvision.io import read_image
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
# import torch

img = read_image(".\\main\\cropPhoto\\cropPhotocrop_4.jpg")

# Step 1: Initialize model with the best available weights
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)
model.eval()

# Step 2: Initialize the inference transforms
preprocess = weights.transforms()

# Step 3: Apply inference preprocessing transforms
batch = preprocess(img).unsqueeze(0)

# Step 4: Use the model and print the predicted category
prediction = model(batch).squeeze(0).softmax(0)
class_id = prediction.argmax().item()
score = prediction[class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score:.1f}%")
