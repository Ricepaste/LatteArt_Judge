from src.training.Hebbian_train import Hebbian_SSL_Trainer
import torchvision.models as models

# --- Main Entry Point Example ---
if __name__ == "__main__":
    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.shufflenet_v2_x0_5,
        target_sparsity=0.8
    )
    
    trainer.train(
        num_epochs=100, 
        batch_size=128, 
        init_grow_ratio=0.2, 
        hebbian_freq=50 # 每 50 個 batch 更新一次 Pearson 統計量
    )

    mask_path = "runs/Hebbian_SSL_20260105-143940/best.pt" 
    # init_path = "runs/Hebbian_SSL_2023xxxx/init.pt" # 如果有的話
    
    # trainer.validate_lottery(
    #     mask_source_path=mask_path,
    #     weight_init_path=None, # 設為 None 代表隨機重置 (檢查結構本身是否優秀)
    #     num_epochs=100,
    #     batch_size=128
    # )