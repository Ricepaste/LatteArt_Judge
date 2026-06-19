from src.training.Hebbian_train import Hebbian_SSL_Trainer
import torch.nn as nn
from torchvision import models
import os

import numpy as np
import torch
import random
# --- Main Entry Point Example ---
if __name__ == "__main__":
    
    # 在程式最前端拉起 Seed 管控，確保初始遮罩絕對隨機但可復現
    run_seed = int(os.environ.get("RUN_SEED", "42"))
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)
    torch.backends.cudnn.deterministic = True
    
    # --- Ablation Study Toggles ---
    # 支援透過環境變數動態控制，若為原版 SET 實驗則預設為 False
    is_set_run = os.environ.get("ABLATION_RANDOM_GROWTH", "0") == "1"
    USE_ERK = os.environ.get("USE_ERK", "False" if is_set_run else "True") == "True"
    PROTECT_HIGHWAY = os.environ.get("PROTECT_HIGHWAY", "False") == "True"
    # ------------------------------

    target_sparsity_val = float(os.environ.get("TARGET_SPARSITY", "0.99"))
    dataset_name_val = os.environ.get("TARGET_DATASET", "cifar10")
    num_epochs_val = int(os.environ.get("NUM_EPOCHS", "400"))

    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        target_sparsity=target_sparsity_val,
        use_erk=USE_ERK,
        protect_highway=PROTECT_HIGHWAY,
        dataset_name=dataset_name_val
    )
    
    workers_val = int(os.environ.get("DATALOADER_WORKERS", "8"))
    trainer.train(
        num_epochs=num_epochs_val, # Hebbian V10 Automated Epochs
        batch_size=128, 
        workers=workers_val,
        init_grow_ratio=0.2, 
        hebbian_freq=10 # Hebbian V5: 更頻繁地觀察以對抗噪聲
    )

    mask_path = "runs/Hebbian_SSL_20260105-143940/best.pt" 
    # init_path = "runs/Hebbian_SSL_2023xxxx/init.pt" # 如果有的話
    
    # trainer.validate_lottery(
    #     mask_source_path=mask_path,
    #     weight_init_path=None, # 設為 None 代表隨機重置 (檢查結構本身是否優秀)
    #     num_epochs=100,
    #     batch_size=128
    # )