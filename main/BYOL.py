from json import load
from torchvision.io import read_image
from torchvision.models import EfficientNet_B0_Weights
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import src.training.BYOL_train as BYOL_train

MODE = 1  # mode 1 means training, mode 2 means testing


def main():
    if MODE == 1:
        byol_model = BYOL_train.BYOL_Model()
        byol_model.train(
            num_epochs=20,
            batch_size=40,
            dataset_dir=".\\LabelTool\\Unlabeled_photo",
        )


if __name__ == "__main__":
    main()
