from json import load
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import torchvision.models as models

import src.training.Semi_Sparse_SimSiam_train as ADS_SSL_training_flow

# 設置共用的訓練參數
common_train_params = {
    "num_epochs": 100,  # 可以設長一點試試
    "batch_size": 128,
    "workers": 0,  # 根據你的設備調整
    "dataset_dir": ".\\LabelTool",  # 你的資料集路徑
    "rigl_dense_allocation": 0.6,  # RigL 稀疏度 (90% sparse)
    "rigl_delta": 100,
    "rigl_alpha": 0.3,
    "consistency_lambda": 0.1,
}


# TODO: 刪除無用的dataset_dir參數
def main():
    model_trainer = ADS_SSL_training_flow.ADS_SSL_Model(
        pretrained_model_class=models.shufflenet_v2_x0_5,
        base_lr=0.05,

    )
    model_trainer.train(
        num_epochs=100,
        batch_size=128,
        workers=0,
        dataset_dir="D:\\Dataset\\train",
        lambda_val=1e-2,
        mask_update_freq=100,
        alpha_initial=1.0,
        alpha_final=50.0,
        lr_mask=0.01,
        momentum=0.5
    )


if __name__ == "__main__":
    main()