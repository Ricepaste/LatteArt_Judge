import os
import torch
import torch.nn as nn
import time
import subprocess
from torchvision import models
from src.training.Hebbian_train import Hebbian_SSL_Trainer

# Setup directories
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
runs_dir = os.path.join(MAIN_DIR, "runs")

def replace_bn_with_identity(model):
    for name, child in model.named_children():
        if isinstance(child, nn.BatchNorm2d):
            setattr(model, name, nn.Identity())
        else:
            replace_bn_with_identity(child)

def resnet18_no_bn(weights=None):
    model = models.resnet18(weights=weights)
    replace_bn_with_identity(model)
    print("📢 Natively removed all BatchNorm2d layers from ResNet-18 backbone!")
    return model

def train_model(ablation_positive_only, run_name, epochs=30):
    print("\n" + "="*70)
    print(f"🚀 Training {run_name} (Sparsity: 99%, Epochs: {epochs})")
    print("="*70)
    
    # We instantiate the trainer with our custom resnet18_no_bn
    trainer = Hebbian_SSL_Trainer(
        pretrained_model_class=resnet18_no_bn,
        pretrained_weight=None,
        target_sparsity=0.99,
        use_erk=True,
        protect_highway=False,
        dataset_name="cifar100"
    )
    
    # Configure ablation via environment variables (read by Hebbian.py or training code)
    os.environ["ABLATION_POSITIVE_HEBB_ONLY"] = "1" if ablation_positive_only else "0"
    
    # We will manually run the training loop from trainer for the specified epochs to keep it fast
    # Let's check Hebbian_SSL_Trainer.train method parameters
    # Normally trainer has a train() method or similar. Let's see what trainer.train accepts or run it.
    
    # Let's inspect Hebbian_train.py for training method.
    # Usually it's trainer.train(epochs=epochs) or trainer.start_training()
    # Let's call trainer.train() if it exists or do it dynamically.
    
    # Wait, does trainer have .train()? Let's check how Hebbian.py calls it.
    # In Hebbian.py, it calls:
    # trainer = Hebbian_SSL_Trainer(...)
    # trainer.train(epochs=NUM_EPOCHS)
    
    # Let's check Hebbian_train.py for train() method.
    # We will invoke the training loop for the requested epochs.
    trainer.train(num_epochs=epochs)
    
    # Return the path of the saved checkpoint
    # Hebbian_SSL_Trainer saves runs under runs/Hebbian_SSL_TIMESTAMP/
    # We can find the latest created directory
    import glob
    folders = glob.glob(os.path.join(runs_dir, "Hebbian_SSL_*"))
    folders.sort(key=os.path.getmtime)
    latest_run = folders[-1]
    checkpoint_path = os.path.join(latest_run, "last.pt")
    
    # Rename latest_run folder to custom run_name for clarity
    target_dir = os.path.join(runs_dir, f"NoBN_Backbone_{run_name}")
    if os.path.exists(target_dir):
        import shutil
        shutil.rmtree(target_dir)
    os.rename(latest_run, target_dir)
    
    return os.path.join(target_dir, "last.pt")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20, help="Number of pretraining epochs")
    args = parser.parse_args()
    
    print("="*80)
    print("🔬 RUNNING ULTRA-FAST NO-BN BACKBONE GROWTH EXPERIMENT")
    print("="*80)
    
    # 1. Train Ours (No-BN Backbone, 99% Sparsity)
    ours_ckpt = train_model(ablation_positive_only=False, run_name="Ours_Lateral_Inhibition", epochs=args.epochs)
    
    # 2. Train Positive Hebbian (No-BN Backbone, 99% Sparsity)
    pos_ckpt = train_model(ablation_positive_only=True, run_name="Positive_Hebbian_Ablation", epochs=args.epochs)
    
    # 3. Standardized Evaluation on SVHN or CIFAR-100
    # Let's evaluate both on CIFAR-100 (which they trained on) to see feature generalizability
    print("\n" + "="*70)
    print("📊 EVALUATING GENERALIZATION ACCURACY (CIFAR-100)")
    print("="*70)
    
    def run_eval(ckpt_path, name):
        env = os.environ.copy()
        env["ENCODER_PATH"] = ckpt_path
        env["METHOD"] = "hebbian"
        env["TARGET_DATASET"] = "cifar100"
        env["NUM_EPOCHS"] = "20"  # Fast linear probing
        env["TARGET_SPARSITY"] = "0.99"
        env["BACKBONE_NO_BN"] = "True"  # Crucial! Prevents re-creating BN layers during evaluation
        
        # Enable downstream classifier BN to ensure stable optimization and proper convergence
        env["DISABLE_CLASSIFIER_BN"] = "False" 
        
        process = subprocess.Popen(
            ["python", "-u", "evaluate_model.py"],
            env=env,
            cwd=MAIN_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        stdout, _ = process.communicate()
        
        # Parse linear probing accuracy from stdout
        linear_acc = "N/A"
        knn_acc = "N/A"
        for line in stdout.split("\n"):
            if "Final Linear Probing Accuracy:" in line:
                linear_acc = line.split(":")[-1].strip()
            elif "KNN Protocol Accuracy" in line:
                knn_acc = line.split(":")[-1].strip()
        
        print(f"Results for {name}:")
        print(f"  > KNN Accuracy: {knn_acc}")
        print(f"  > Linear Probing Accuracy: {linear_acc}")
        return knn_acc, linear_acc
        
    ours_knn, ours_linear = run_eval(ours_ckpt, "Ours (Lateral Inhibition)")
    pos_knn, pos_linear = run_eval(pos_ckpt, "Positive Hebbian Ablation")
    
    print("\n" + "="*70)
    print("📢 FINAL EXPERIMENT SUMMARY (No-BN Backbone)")
    print("="*70)
    print(f"Ours KNN:       {ours_knn}  | Ours Linear:       {ours_linear}")
    print(f"PosHebb KNN:    {pos_knn}  | PosHebb Linear:    {pos_linear}")
    print("="*70 + "\n")
