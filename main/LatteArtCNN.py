import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Running Device: ", end="")
print("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.hub.load('ultralytics/yolov5', 'custom',
#                        path='./best.pt').to(device)
model = torch.hub.load('.\\yolov5', 'custom', source='local',
                       path='.\\Crop_Model\\exp62_v3\\weights\\best.pt',
                       force_reload=True).to(device)

imgs = ['.\\main\\inputPhoto\\1.jpg']
results = model(imgs, size=64)
results.print()

results.show()
