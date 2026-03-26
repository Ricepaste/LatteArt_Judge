from json import load
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import torchvision.models as models

import src.training.SimSiam_train as SimSiam_train

# 設置共用的訓練參數 (對標 Hebbian V7)
common_train_params = {
    "num_epochs": 1000, 
    "batch_size": 128,
    "workers": 0,
    "dataset_dir": ".\\LabelTool",
    "rigl_dense_allocation": 0.2,  # 相對應於 Hebbian 的 target_sparsity=0.8
    "rigl_delta": 100,
    "rigl_alpha": 0.3,
    "consistency_lambda": 0.107,
}


# TODO: 刪除無用的dataset_dir參數
def main():
    model_rigl_baseline = SimSiam_train.SimSiam_Model(
        pretrained_model=models.resnet18,
        base_lr=0.03
    )
    # model_rigl_baseline.Lottery_validation(
    #     sparse_model_state_dict_path="./runs/shuffleNet_v05_SimSiam__7/best.pt",
    #     rewind_weight_prams_path="./runs/shuffleNet_v05_SimSiam__7/early.pt",
    # )
    print("\n--- Running Baseline RigL on SimSiam ---")
    model_rigl_baseline.train(
        **common_train_params,
        # rigl_mode="none",  # 設置為 baseline
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

    # model_rigl_consistency = SimSiam_train.SimSiam_Model(base_lr=0.03)
    # print("\n--- Running Consistency RigL on SimSiam ---")
    # model_rigl_consistency.train(
    #     **common_train_params,
    #     rigl_mode="none",  # 設置為 consistency
    #     grad_cache_chunk_size=0,  # 不使用 GradCache
    # )


if __name__ == "__main__":
    main()
