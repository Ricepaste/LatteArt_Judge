# main/Dense.py
from torchvision.io import read_image
from torch.utils.tensorboard import SummaryWriter  # type: ignore
import torchvision.models as models
import os
import numpy as np
import torch
import random
import src.training.SimSiam_train as SimSiam_train

def main():
    # 在程式最前端拉起 Seed 管控，確保初始隨機網路拓樸可控
    run_seed = int(os.environ.get("RUN_SEED", "42"))
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)
    torch.backends.cudnn.deterministic = True
    
    dataset_name_val = os.environ.get("TARGET_DATASET", "cifar10")
    num_epochs_val = int(os.environ.get("NUM_EPOCHS", "400"))
    
    # 設置共用的訓練參數 (對標 Hebbian V7)
    common_train_params = {
        "num_epochs": num_epochs_val, 
        "batch_size": 128,
        "workers": 0,
        "dataset_dir": ".\\LabelTool",
        "dataset_name": dataset_name_val,
        "rigl_dense_allocation": 1.0,  # Fully Dense (100% weights)
        "rigl_delta": 100,
        "rigl_alpha": 0.3,
        "consistency_lambda": 0.107,
    }

    model_dense = SimSiam_train.SimSiam_Model(
        pretrained_model=models.resnet18,
        base_lr=0.03
    )
    
    print("\n--- Running Fully Dense SimSiam Baseline ---")
    model_dense.train(
        **common_train_params,
        rigl_mode="none",  # "none" setting triggers fully dense training
        grad_cache_chunk_size=0,  # 不使用 GradCache
    )

if __name__ == "__main__":
    main()
