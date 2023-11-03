import torch
import cv2

show_img = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Running Device: ", end="")
print("cuda" if torch.cuda.is_available() else "cpu")

# model = torch.hub.load('ultralytics/yolov5', 'custom',
#                        path='./best.pt').to(device)
model = torch.hub.load('.\\yolov5', 'custom', source='local',
                       path='.\\Crop_Model\\exp62_v3\\weights\\best.pt',
                       force_reload=True).to(device)

imgs = ['.\\main\\inputPhoto\\{}.jpg'.format(i)for i in range(5)]
results = model(imgs, size=64)
results.print()

if (show_img):
    results.show()
# print(results.xyxy)

output_index = 0
for img in range(len(imgs)):
    for i in range(results.xyxy[img].cpu().numpy().shape[0]):
        crop_img = cv2.imread(imgs[img])
        x_lt = results.xyxy[img].cpu().numpy()[i][0]
        y_lt = results.xyxy[img].cpu().numpy()[i][1]
        x_rb = results.xyxy[img].cpu().numpy()[i][2]
        y_rb = results.xyxy[img].cpu().numpy()[i][3]
        crop_img = crop_img[int(y_lt):int(y_rb), int(x_lt):int(x_rb)]

        if (show_img):
            cv2.imshow("cropped", crop_img)
            cv2.waitKey(0)

        output_index += 1
        path = '.\\main\\cropPhoto\\cropPhotocrop_{}.jpg'.format(output_index)
        cv2.imwrite(path, crop_img)
