import torch
import cv2


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

for img in imgs:
    crop_img = cv2.imread(img)
    x_lt = results.xyxy[0].cpu().numpy()[0][0]
    y_lt = results.xyxy[0].cpu().numpy()[0][1]
    x_rb = results.xyxy[0].cpu().numpy()[0][2]
    y_rb = results.xyxy[0].cpu().numpy()[0][3]
    crop_img = crop_img[int(y_lt):int(y_rb), int(x_lt):int(x_rb)]
    cv2.imshow("cropped", crop_img)
    cv2.waitKey(0)
