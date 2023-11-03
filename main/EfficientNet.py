import torch
import os
from torchvision import datasets, transforms
import torch.optim as optim
from torchvision.io import read_image
import torchvision.models as models
from torchvision.models import EfficientNet_B1_Weights


def main():
    # 訓練資料進行資料增補，驗證資料不需要
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),  # 資料增補
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 準備資料集匯入器
    # 使用 ImageFolder 可方便轉換為 dataset
    data_dir = '.\\main\\cropPhoto'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                  batch_size=4,
                                                  shuffle=True,
                                                  num_workers=0)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)

    xx = input('press any key to continue')

    for i, x in dataloaders['train']:
        print(i.shape)
        print(x)

    # # 顯示一批資料
    # print(classes)

    # Step 1: Initialize model with the best available weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = EfficientNet_B1_Weights.DEFAULT
    model = models.efficientnet_b1(weights=weights)

    # Step 2: Initialize the inference transforms
    # 資料強化還沒有做，需要研究原轉換函數內容
    preprocess = weights.transforms()
    # print(preprocess)

    # freeze all the parameters in the old model
    for param in model.parameters():
        param.requires_grad = False

    # print the model structure to see the last layer
    print('old top layer:', model._modules['classifier'], sep='\n')

    # change the last layer of the model to fit our problem
    model._modules['classifier'] = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=False),
        torch.nn.Linear(1280, 1))

    print('new top layer for transfer learning',
          model._modules['classifier'], sep='\n')

    # 使用MSE作為損失函數，reduction='mean'表示計算均值，功能不明
    criterion = torch.nn.MSELoss(reduction='mean')

    # 定義優化器為隨機梯度下降，學習率為0.001，動量為0.9
    optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model = model.to(device)

    model.train()

    # ------------------------------train done------------------------------------

    # start evaluating the model
    model.eval()

    # Step 3: Apply inference preprocessing transforms
    img = read_image(".\\main\\cropPhoto\\cropPhotocrop_2.jpg")
    batch = preprocess(img).unsqueeze(0).to(device)

    # Step 4: Use the model and print the predicted category
    # prediction = model(batch).squeeze(0).softmax(0)
    prediction = model(batch).squeeze(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

    print(prediction.detach().cpu().numpy())
    for i in range(prediction.detach().cpu().numpy().shape[0]):
        print(weights.meta["categories"][i])


if __name__ == "__main__":
    main()
