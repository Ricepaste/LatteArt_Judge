from src.training.Hebbian_train import Hebbian_SSL_Trainer
import torchvision.models as models

# --- Main Entry Point Example ---
if __name__ == "__main__":
    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=models.resnet18,
        target_sparsity=0.8
    )
    
    trainer.train(
        num_epochs=1000, # Hebbian V7: 延長生物探索期
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