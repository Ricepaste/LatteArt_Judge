from json import load
from TrainModel import *
from torchvision.io import read_image
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import Siamese_Model

# TODO label標準化、不要每次都讀取資料集、損失函數重新設計、神經元增加、圖片先分類

WORKERS = 0
LR = 0.001
MOMENTUM = 0.1
BATCH_SIZE = 8
EPOCHS = 100
LOAD_MODEL = True
LOAD_MODEL_PATH = ".\\EFN_Model\\best_mini.pt"
MODE = 1  # mode 1 means training, mode 2 means testing
GRAY_VISION = True
GRAY_VISION_PREVIEW = True
TRAIN_EFN = False


def main():
    if MODE == 1:
        Latte_Model = LatteArtJudge_Model(
            pretrained_model=models.efficientnet_b0,
            pretrained_weight=EfficientNet_B0_Weights.DEFAULT,
            model=Siamese_Model.SNN,
            load_weight="",
            freeze=True,
        )
        Latte_Model.train(
            num_epochs=100,
            train_efn=True,
            batch_size=8,
            workers=0,
            dataset_dir=".\\LabelTool\\backup27",
        )
        Latte_Model.test_one_run()


if __name__ == "__main__":
    main()

"""
# region TODO: 整理程式碼

if LOAD_MODEL:
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))
# start evaluating the model
model.eval()

gray = transforms.Compose(
    [
        transforms.Resize(900),
        # transforms.ColorJitter(contrast=(0.9, 0.9), saturation=(1.2, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Grayscale(num_output_channels=3),
    ]
)

# Step 3: Apply inference preprocessing transforms
if GRAY_VISION:
    img = Image.open(".\\main\\cropPhoto\\test.jpg").convert("RGB")
    open_cv_image = np.array(img)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    open_cv_image = cv2.bilateralFilter(open_cv_image, 20, 50, 100)
    img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    batch = gray(img).unsqueeze(0).to(device)  # type: ignore
else:
    img = read_image(".\\main\\cropPhoto\\test.jpg")
    batch = preprocess(img).unsqueeze(0).to(device)

if GRAY_VISION_PREVIEW:
    tt = transforms.ToPILImage()
    imgg = tt(gray(img))
    imgg.show()

# Step 4: Use the model and print the predicted category
# prediction = model(batch).squeeze(0).softmax(0)
prediction = model(batch).squeeze(0)
# class_id = prediction.argmax().item()
# score = prediction[class_id].item()
# category_name = weights.meta["categories"][class_id]
# print(f"{category_name}: {100 * score:.1f}%")

print(
    "\nThe Score of TEST photo is: {}point".format(prediction.detach().cpu().numpy()[0])
)
# for i in range(prediction.detach().cpu().numpy().shape[0]):
#     print(weights.meta["categories"][i])
"""
