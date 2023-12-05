from TrainModel import *

from torchvision.io import read_image
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
import torch.optim as optim
from torch.optim import lr_scheduler


# TODO label標準化、不要每次都讀取資料集、損失函數重新設計、照片排序比分、神經元增加、圖片先分類

WORKERS = 0
LR = 0.001
MOMENTUM = 0.1
BATCH_SIZE = 8
EPOCHS = 100
LOAD_MODEL = True
LOAD_MODEL_PATH = '.\\EFN_Model\\best_mini.pt'
MODE = 'test'  # train or test
GRAY_VISION = True
GRAY_VISION_PREVIEW = True
TRAIN_EFN = False


# -------------------------main---------------------------

# Step 1: Initialize model with the best available weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weights)

# Step 2: Initialize the inference transforms
# 資料強化還沒有做，需要研究原轉換函數內容
preprocess = weights.transforms()
print(preprocess)

# 訓練資料進行資料增補，驗證資料不需要


# freeze all the parameters in the old model
for param in model.parameters():
    param.requires_grad = False

# print the model structure to see the last layer
# print('old top layer:', model._modules['classifier'], sep='\n')

# # change the last layer of the b1 model to fit our problem
# model._modules['classifier'] = torch.nn.Sequential(
#     # torch.nn.FeatureAlphaDropout(p=0.2, inplace=False),
#     torch.nn.Linear(1280, 400),
#     # torch.nn.Sigmoid(),
#     torch.nn.LeakyReLU(),
#     torch.nn.Dropout(p=0.2, inplace=False),
#     torch.nn.Linear(400, 1),
#     # torch.nn.Linear(1280, 1),
# )

# change the last layer of the b3 model to fit our problem
model._modules['classifier'] = torch.nn.Sequential(
    # torch.nn.FeatureAlphaDropout(p=0.2, inplace=False),
    torch.nn.Linear(1280, 20000),
    torch.nn.Sigmoid(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(20000, 5000),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(5000, 100),
    torch.nn.LeakyReLU(),
    torch.nn.Dropout(p=0.2, inplace=False),
    torch.nn.Linear(100, 1),
    # torch.nn.Linear(1280, 1),
)

# print('new top layer for transfer learning',
#       model._modules['classifier'], sep='\n')

# 使用MSE作為損失函數，reduction='mean'表示計算均值，功能不明
# criterion = torch.nn.MSELoss(reduction='mean')

# 使用MarginRankingLoss作為損失函數
criterion = torch.nn.MarginRankingLoss(margin=0.5, reduction='mean')

# # 使用AdamW作為优化器
# optimizer = optim.AdamW(model.parameters())

# # 使用ReduceLROnPlateau作為学习率调整器
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer, mode='min', factor=0.1, patience=5)

# 定義優化器為隨機梯度下降，學習率為0.001，動量為0.9
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)

model = model.to(device)

# 每7個執行週期，學習率降 0.1
scheduler = lr_scheduler.StepLR(
    optimizer, step_size=7, gamma=0.1)

# exp_lr_scheduler = lr_scheduler.StepLR(
#     optimizer, step_size=7, gamma=0.1, last_epoch=)

if (LOAD_MODEL):
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))

if MODE == 'train':
    # model = train_model(model, criterion, optimizer,
    #                     exp_lr_scheduler, num_epochs=EPOCHS)
    initialize(BATCH_SIZE, WORKERS)
    model = train_model(model, criterion, optimizer, scheduler, device,
                        num_epochs=EPOCHS, TRAIN_EFN=TRAIN_EFN, BATCH_SIZE=BATCH_SIZE, MODE=MODE)

    # files = os.listdir('.\\EFN_Model')
    # i = 0
    # name = "best.pt"
    # while name in files:
    #     i += 1
    #     name = "best{}.pt".format(i)
    # torch.save(model.state_dict(), '.\\EFN_Model\\{}'.format(name))


# ------------------------------train done--------------------------------

if (LOAD_MODEL):
    model.load_state_dict(torch.load(LOAD_MODEL_PATH))
# start evaluating the model
model.eval()

gray = transforms.Compose([
    transforms.Resize(900),
    # transforms.ColorJitter(contrast=(0.9, 0.9), saturation=(1.2, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    transforms.Grayscale(num_output_channels=3),
])

# Step 3: Apply inference preprocessing transforms
if GRAY_VISION:
    img = Image.open(".\\main\\cropPhoto\\test.jpg").convert('RGB')
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

print('\nThe Score of TEST photo is: {}point'.format(
    prediction.detach().cpu().numpy()[0]))
# for i in range(prediction.detach().cpu().numpy().shape[0]):
#     print(weights.meta["categories"][i])


# if __name__ == "__main__":
#     main()
