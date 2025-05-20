from json import load
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter  # type: ignore

import src.training.SimSiam_train as SimSiam_train

# 設置共用的訓練參數
common_train_params = {
    "num_epochs": 100,  # 可以設長一點試試
    "batch_size": 40,
    "workers": 0,  # 根據你的設備調整
    "dataset_dir": ".\\LabelTool",  # 你的資料集路徑
    "rigl_dense_allocation": 0.15,  # RigL 稀疏度 (90% sparse)
    "rigl_delta": 100,
    "rigl_alpha": 0.3,
    "consistency_lambda": 1,
}


# TODO: 刪除無用的dataset_dir參數
def main():
    model_rigl_baseline = SimSiam_train.SimSiam_Model(base_lr=0.03)
    print("\n--- Running Baseline RigL on SimSiam ---")
    model_rigl_baseline.train(
        **common_train_params,
        rigl_mode="baseline",  # 設置為 baseline
        grad_cache_chunk_size=0,  # 不使用 GradCache
    )

    # model_rigl_consistency = SimSiam_train.SimSiam_Model(base_lr=0.03)
    # print("\n--- Running Consistency RigL on SimSiam ---")
    # model_rigl_consistency.train(
    #     **common_train_params,
    #     rigl_mode="consistency",  # 設置為 consistency
    #     grad_cache_chunk_size=0,  # 不使用 GradCache
    # )


if __name__ == "__main__":
    main()
