from PIL import Image
import os

input_folder = "LabelTool/Unlabeled_photo/"
output_folder = "LabelTool/Unlabeled_photo/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith((".jpg", ".jpeg", ".png", ".bmp")):
        img = Image.open(os.path.join(input_folder, filename))
        img = img.resize((224, 224), Image.BILINEAR)  # type: ignore # 使用 BICUBIC 替代 LANCZOS
        output_path = os.path.join(output_folder, filename)
        img.save(output_path, quality=85)  # quality 可以调整，数值越低画质越低
print("Done!")
