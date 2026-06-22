# main/evaluate_segmentation.py
import os
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torchvision.transforms.functional as TF
from PIL import Image

# Fix seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# Custom transform wrapper for joint segmentation dataset transformation
class JointTransform:
    def __init__(self, size=(256, 256), is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img, mask):
        # Resize
        img = TF.resize(img, self.size)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if self.is_train:
            # Random horizontal flip
            if random.random() > 0.5:
                img = TF.hflip(img)
                mask = TF.hflip(mask)

        # Convert image to tensor and normalize (standard ImageNet normalization)
        img = TF.to_tensor(img)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Convert mask target to tensor (keep as long indices, ignore boundary label 255)
        mask = torch.as_tensor(np.array(mask), dtype=torch.long)
        return img, mask

# Simple FCN head for semantic segmentation
class FCNHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x

class SparseSegmentationModel(nn.Module):
    def __init__(self, encoder, num_classes=21):
        super().__init__()
        self.encoder = encoder
        self.head = FCNHead(512, num_classes)

    def forward(self, x):
        input_shape = x.shape[-2:]
        # Get output from ResNet18 encoder (shape: B, 512, H/32, W/32)
        features = self.encoder(x)
        out = self.head(features)
        # Upsample back to original image size
        out = F.interpolate(out, size=input_shape, mode='bilinear', align_corners=True)
        return out

def compute_metrics(preds, targets, num_classes=21):
    # Ignore boundary class 255
    valid_mask = (targets != 255)
    
    # Pixel Accuracy
    correct_pixels = (preds[valid_mask] == targets[valid_mask]).sum().item()
    total_pixels = valid_mask.sum().item()
    pixel_acc = correct_pixels / (total_pixels + 1e-8)
    
    # Mean IoU
    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls) & valid_mask
        target_cls = (targets == cls) & valid_mask
        
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        
        # Only compute IoU for classes present in the ground truth target
        if target_cls.sum().item() == 0:
            continue
            
        iou = intersection / (union + 1e-8)
        ious.append(iou)
        
    mean_iou = np.mean(ious) if len(ious) > 0 else 0.0
    return pixel_acc, mean_iou

def load_pretrain_encoder(method, path, sparsity, use_erk, protect_highway, device):
    if path.lower() in ["imagenet", "official"]:
        print("🌍 Loading official ImageNet pre-trained ResNet18 weights...")
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Create a sequential encoder equivalent to our module's encoder structure
        encoder = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4
        ).to(device)
        return encoder

    print(f"Initializing {method.upper()} model backbone...")
    if method.lower() == "hebbian":
        from src.training.Hebbian_train import Hebbian_SSL_Trainer
        dummy_trainer = Hebbian_SSL_Trainer(
            pretrained_model_class=models.resnet18,
            pretrained_weight=None,
            target_sparsity=sparsity,
            use_erk=use_erk,
            protect_highway=protect_highway
        )
        model = dummy_trainer.model.to(device)
    else:
        import src.training.SimSiam_train as SimSiam_train
        dummy_trainer = SimSiam_train.SimSiam_Model(
            pretrained_model=models.resnet18,
            pretrained_weight=None
        )
        model = dummy_trainer.model.to(device)

    print(f"Loading checkpoint weights from {path}...")
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    
    if hasattr(model, 'set_hebbian_enable'):
        model.set_hebbian_enable(False)
        
    encoder = model.encoder.to(device)
    return encoder

def main():
    parser = argparse.ArgumentParser(description="Evaluate Pre-trained Backbones on Pascal VOC 2012 Segmentation")
    parser.add_argument("--method", type=str, default="hebbian", choices=["hebbian", "rigl", "imagenet"], help="Pre-training method")
    parser.add_argument("--encoder_path", type=str, required=True, help="Path to pre-trained last.pt checkpoint")
    parser.add_argument("--sparsity", type=float, default=0.9, help="Target sparsity (default: 0.9)")
    parser.add_argument("--use_erk", type=str, default="True", help="Whether pre-trained model used ERK")
    parser.add_argument("--protect_highway", type=str, default="False", help="Whether pre-trained model protected residual highway")
    parser.add_argument("--dataset_dir", type=str, default="main/data/voc", help="Pascal VOC dataset root directory")
    parser.add_argument("--download", action="store_true", help="Download dataset if not present")
    parser.add_argument("--mode", type=str, default="probing", choices=["probing", "finetuning"], help="Evaluation mode")
    parser.add_argument("--epochs", type=int, default=30, help="Number of semantic segmentation training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--workers", type=int, default=4, help="Dataloader workers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 從權重路徑提取資料夾名稱以避免檔名衝突並方便識別
    log_dir = "main/ablation_logs"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_folder = "unknown_run"
    if args.encoder_path:
        norm_path = os.path.normpath(args.encoder_path)
        parts = norm_path.split(os.sep)
        if len(parts) >= 2:
            checkpoint_folder = parts[-2] if parts[-1].endswith(".pt") else parts[-1]
            
    log_file = os.path.join(log_dir, f"segmentation_{checkpoint_folder}_{args.mode}.log")

    use_erk_bool = args.use_erk.lower() == "true"
    protect_highway_bool = args.protect_highway.lower() == "true"

    # Set up datasets
    train_transform = JointTransform(size=(256, 256), is_train=True)
    val_transform = JointTransform(size=(256, 256), is_train=False)

    try:
        train_dataset = models.segmentation.VOCSegmentation(
            root=args.dataset_dir, year="2012", image_set="train",
            download=args.download, transforms=train_transform
        )
        val_dataset = models.segmentation.VOCSegmentation(
            root=args.dataset_dir, year="2012", image_set="val",
            download=args.download, transforms=val_transform
        )
    except Exception as e:
        print(f"❌ Error loading Pascal VOC 2012 dataset: {e}")
        print("💡 Make sure the dataset is downloaded or set --download flag if internet is available.")
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    print("="*60)
    print(f"🚀 Pascal VOC 2012 Semantic Segmentation Evaluation")
    print(f"Method: {args.method.upper()} | Mode: {args.mode.upper()}")
    print(f"Checkpoint: {args.encoder_path}")
    print(f"Target Sparsity: {args.sparsity * 100:.1f}%")
    print(f"Training Settings: Epochs={args.epochs}, LR={args.lr}, BS={args.batch_size}")
    print(f"Log File: {log_file}")
    print("="*60)

    # Initialize model
    encoder = load_pretrain_encoder(args.method, args.encoder_path, args.sparsity, use_erk_bool, protect_highway_bool, device)
    model = SparseSegmentationModel(encoder, num_classes=21).to(device)

    # Freeze or unfreeze encoder based on mode
    if args.mode == "probing":
        print("🔒 Mode: PROBING (Freezing Backbone, training segmentation head only)")
        for param in model.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        print("🔓 Mode: FINETUNING (Training both Backbone and segmentation head)")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_miou = 0.0

    with open(log_file, "w") as f_log:
        f_log.write(f"=== Semantic Segmentation Evaluation Log ===\n")
        f_log.write(f"Method: {args.method}\nMode: {args.mode}\nSparsity: {args.sparsity}\nEncoder Path: {args.encoder_path}\n\n")

        for epoch in range(args.epochs):
            model.train()
            total_loss = 0.0
            
            for imgs, masks in train_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                
                optimizer.zero_grad()
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                
                # In Fine-tuning mode, enforce the sparse masks on the encoder weights after each optimization step
                if args.mode == "finetuning":
                    with torch.no_grad():
                        for m in model.encoder.modules():
                            if hasattr(m, 'mask') and m.mask is not None:
                                m.layer.weight.data *= m.mask
                
                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)
            
            # Validation
            model.eval()
            val_pixel_accs = []
            val_mious = []
            
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs, masks = imgs.to(device), masks.to(device)
                    outputs = model(imgs)
                    
                    preds = torch.argmax(outputs, dim=1)
                    
                    pixel_acc, miou = compute_metrics(preds, masks, num_classes=21)
                    val_pixel_accs.append(pixel_acc)
                    val_mious.append(miou)
            
            epoch_miou = np.mean(val_mious)
            epoch_acc = np.mean(val_pixel_accs)
            scheduler.step()

            log_str = f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Pixel Acc: {epoch_acc:.4f} | Val mIoU: {epoch_miou:.4f}"
            print(log_str)
            f_log.write(log_str + "\n")
            f_log.flush()

            if epoch_miou > best_miou:
                best_miou = epoch_miou
                best_log = f"⭐ New Best mIoU: {best_miou:.4f} at epoch {epoch+1}"
                print(best_log)
                f_log.write(best_log + "\n")
                f_log.flush()

        summary_str = f"\n=== Final Summary ===\nBest Validation mIoU: {best_miou:.4f}\n"
        print(summary_str)
        f_log.write(summary_str)

    print(f"✅ Evaluation completed! Full log written to: {log_file}")

if __name__ == "__main__":
    main()
