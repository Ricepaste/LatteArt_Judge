from src.training.Hebbian_train import Hebbian_SSL_Trainer
import torch.nn as nn
from torchvision import models

# --- Main Entry Point Example ---
if __name__ == "__main__":
    
    # --- Ablation Study Toggles ---
    # 設為 True : 啟用 V6 特性 (1000 Epoch 可達 ~80%)
    # 設為 False: 退回 V5 特性 (用來證明 V6 的架構貢獻)
    USE_ERK = True          
    PROTECT_HIGHWAY = False 
    # ------------------------------

    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        target_sparsity=0.99,
        use_erk=USE_ERK,
        protect_highway=PROTECT_HIGHWAY
    )
    
    trainer.train(
        num_epochs=400, # Hebbian V7: 延長生物探索期
        batch_size=128, 
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