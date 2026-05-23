# main/Random.py
from src.training.SET_train import SET_SSL_Trainer
import torchvision.models as models
import os
import numpy as np
import torch
import random

if __name__ == "__main__":
    # 在程式最前端拉起 Seed 管控，確保初始隨機網路拓樸可控
    run_seed = int(os.environ.get("RUN_SEED", "42"))
    torch.manual_seed(run_seed)
    torch.cuda.manual_seed_all(run_seed)
    np.random.seed(run_seed)
    random.seed(run_seed)
    torch.backends.cudnn.deterministic = True
    
    target_sparsity_val = float(os.environ.get("TARGET_SPARSITY", "0.99"))
    dataset_name_val = os.environ.get("TARGET_DATASET", "cifar10")
    num_epochs_val = int(os.environ.get("NUM_EPOCHS", "400"))
    
    trainer = SET_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        target_sparsity=target_sparsity_val,
        dataset_name=dataset_name_val
    )
    
    print("\n--- Running Random Pruning SimSiam (SET) ---")
    trainer.train(
        num_epochs=num_epochs_val, 
        batch_size=128, 
        init_grow_ratio=0.2, 
        hebbian_freq=10 # 與 Hebbian 頻率對齊
    )
