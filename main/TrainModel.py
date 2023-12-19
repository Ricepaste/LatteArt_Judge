from LatteDataset import *

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import time
import copy
import torch


def initialize(BATCH_SIZE, WORKERS):
    global data_transforms, dataloaders

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(800, scale=(0.8, 1)),  # 資料增補 224
            transforms.Resize(900),
            transforms.ColorJitter(contrast=(0.5, 0.8),  # type: ignore
                                   saturation=(1.2, 1.5)),  # type: ignore
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=10),
        ]),
        'val': transforms.Compose([
            # transforms.Resize((255, 255)),
            transforms.Resize(900),
            transforms.CenterCrop(800),
            transforms.Resize(900),
            # transforms.ColorJitter(contrast=(0.5, 0.8), saturation=(1.2, 1.5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
            transforms.Grayscale(num_output_channels=3),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.RandomRotation(degrees=10),
            # transforms.RandomAffine(degrees=10),
        ]),
    }

    # 準備資料集匯入器
    # 使用 ImageFolder 可方便轉換為 dataset
    data_dir = '.\\LabelTool'
    # image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
    #                                           data_transforms[x])
    #                   for x in ['train', 'val']}
    image_datasets = {x: TonyLatteDataset(os.path.join(data_dir, x),
                                          data_transforms[x], x)
                      for x in ['train', 'val']}
    # image_datasets = {x: TonyLatteDataset(os.path.join(data_dir, x),
    #                                       preprocess)
    #                   for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x],  # type: ignore
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  num_workers=WORKERS)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print(dataset_sizes)

    for i, x in dataloaders['train']:
        print(i.shape)
        print(x)
        print(x.data)


def train_model(model, criterion, optimizer, scheduler, device, num_epochs=25, TRAIN_EFN=False, BATCH_SIZE=8, MODE='train'):
    if MODE == 'train':
        files = os.listdir('.\\runs')
        i = 0
        name = "efficientnet_b1"
        while name in files:
            i += 1
            name = "efficientnet_b1_{}".format(i)
        writer = SummaryWriter('runs\\{}'.format(name))

    since = time.time()
    writer.add_graph(model, torch.zeros(  # type: ignore
        1, 3, 800, 800).to(device))

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # val_index = 0
    # best_loss = float('inf')

    dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in [
        'train', 'val']}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        distribution = []
        distribution_val = []

        # epoch 10 之後開始訓練原模型
        if (epoch == 10 and TRAIN_EFN):
            for param in model.parameters():
                param.requires_grad = True

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if (epoch != 0 and epoch % 10 == 0 and phase == 'train'):
                dataloaders[phase].dataset.restratify()
                dataset_sizes = {phase: len(dataloaders[phase].dataset) for phase in [
                    'train', 'val']}
                print(dataset_sizes['train'])

            # 逐批訓練或驗證
            for inputs, labels in dataloaders[phase]:
                # inputs1 = inputs[:BATCH_SIZE/2]
                # inputs2 = inputs[BATCH_SIZE/2:]
                labels1 = labels[:BATCH_SIZE//2]
                labels2 = labels[BATCH_SIZE//2:]
                if (labels1.shape[0] != labels2.shape[0]):
                    print('labels1.size != labels2.size')
                    continue
                rank_labels = (labels1 > labels2).to(int).to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs1 = inputs1.to(device)
                # inputs2 = inputs2.to(device)
                # labels1 = labels1.to(device)
                # labels2 = labels2.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # 訓練時需要梯度下降
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    outputs = outputs.float()

                    outputs = outputs.squeeze(1)
                    labels = labels.float()

                    outputs1 = outputs[:BATCH_SIZE//2]
                    outputs2 = outputs[BATCH_SIZE//2:]
                    # loss = criterion(outputs, labels)
                    loss = criterion(outputs1, outputs2, rank_labels)

                    # print(outputs, labels)
                    # print('loss:\n', loss)
                    if (loss.item() > 1000):
                        print('loss:', loss)
                    print(outputs1)
                    print(labels)

                    # 訓練時需要 backward + optimize
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 統計損失
                running_loss += loss.item() * inputs.size(0)
                # print(loss, inputs.size(0))
                # 統計正確率
                edge_up = outputs <= (labels.data + 0.5)
                edge_down = outputs > (labels.data - 0.5)

                # 輸出label的分布範圍，確認是否在猜測期望值
                # for i in range(labels.data.shape[0]):

                #     writer.add_scalar('training/output_labels', outputs[i],  # type: ignore
                #                       val_index)
                #     val_index += 1
                # print(outputs.detach().cpu().numpy().astype(float))
                if phase == 'train':
                    distribution.append(
                        outputs.detach().cpu().numpy().astype(float).tolist())
                elif phase == 'val':
                    distribution_val.append(
                        outputs.detach().cpu().numpy().astype(float).tolist())

                # print(labels.data)
                # print(edge_down)
                for i in range(len(edge_up)):
                    if edge_up[i] and edge_down[i]:
                        running_corrects += 1

            if phase == 'train':
                # print(distribution)

                distribution = [y for x in distribution for y in x]
                # print(distribution)
                writer.add_histogram(  # type: ignore
                    'training/output_distribution', np.array(distribution), epoch)
            elif phase == 'val':
                # print(distribution_val)

                distribution_val = [y for x in distribution_val for y in x]
                # print(distribution_val)
                writer.add_histogram(  # type: ignore
                    'validation/output_distribution', np.array(distribution_val), epoch)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = float(running_corrects) / dataset_sizes[phase]

            if phase == 'train':
                writer.add_scalar('training/loss', epoch_loss,  # type: ignore
                                  epoch)
                writer.add_scalar('training/accuracy',  # type: ignore
                                  epoch_acc, epoch)
            elif phase == 'val':
                writer.add_scalar('validation/loss',  # type: ignore
                                  epoch_loss, epoch)
                writer.add_scalar('validation/accuracy',  # type: ignore
                                  epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # 如果是評估階段，且準確率創新高即存入 best_model_wts
            # if phase == 'val' and\
            #         (epoch_acc >= best_acc or epoch_loss <= best_loss):
            if phase == 'val' and (epoch_acc >= best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        (time_elapsed // 60), (time_elapsed % 60)))
    print(f'Best val Acc: {best_acc:4f}')

    # 存最後權重
    files = os.listdir('.\\runs')
    i = 0
    old_name = None
    name = "efficientnet_b1"
    while name in files:
        old_name = name
        i += 1
        name = "efficientnet_b1_{}".format(i)
    torch.save(model.state_dict(), '.\\runs\\{}\\last.pt'.format(old_name))

    # 載入最佳模型
    model.load_state_dict(best_model_wts)

    # 存最佳權重
    files = os.listdir('.\\runs')
    i = 0
    name = "efficientnet_b1"
    while name in files:
        old_name = name
        i += 1
        name = "efficientnet_b1_{}".format(i)
    torch.save(model.state_dict(), '.\\runs\\{}\\best.pt'.format(old_name))

    writer.flush()  # type: ignore
    writer.close()  # type: ignore

    return model
