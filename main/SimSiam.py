from json import load
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import src.training.SimSiam_train as SimSiam_train

MODE = 1  # mode 1 means training, mode 2 means testing


# TODO: 刪除無用的dataset_dir參數
def main():
    if MODE == 1:
        simsiam_model = SimSiam_train.SimSiam_Model(load_weight="")
        simsiam_model.train(
            num_epochs=10,
            batch_size=40,
            grad_cache_chunk_size=4,
            workers=4,
            dataset_dir=".\\LabelTool\\Unlabeled_photo",
        )


if __name__ == "__main__":
    main()
