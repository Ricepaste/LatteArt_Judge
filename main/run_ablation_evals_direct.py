# main/run_ablation_evals_direct.py
import os
import glob
import re
import subprocess
import sys
import time

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(MAIN_DIR, "runs")
LOG_DIR = os.path.join(MAIN_DIR, "ablation_logs")
SUMMARY_CSV = os.path.join(LOG_DIR, "direct_eval_summary.csv")

FOLDERS = [
    "Hebbian_SSL_20260617-182647",
    "Hebbian_SSL_20260617-183925",
    "Hebbian_SSL_20260617-184434",
    "Hebbian_SSL_20260617-185915",
    "Hebbian_SSL_20260617-191027",
    "Hebbian_SSL_20260617-191028",
    "Hebbian_SSL_20260617-191029",
    "Hebbian_SSL_20260617-191858",
    "Hebbian_SSL_20260618-011550"
]

def get_checkpoint_sparsity(checkpoint_path):
    try:
        import torch
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        total_elements = 0
        active_elements = 0
        for k, v in state_dict.items():
            if "mask" in k:
                total_elements += v.numel()
                active_elements += v.sum().item()
        if total_elements > 0:
            return 1.0 - (active_elements / total_elements)
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
    return None

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("============================================================")
    print("🚀 Direct Evaluation Script Started")
    print("============================================================")
    
    # Initialize Summary CSV
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w") as f:
            f.write("Run_Folder,Target_Sparsity,Computed_Sparsity,KNN_Acc,Linear_Acc\n")
            
    for folder_name in FOLDERS:
        folder_path = os.path.join(RUNS_DIR, folder_name)
        checkpoint_path = os.path.join(folder_path, "last.pt")
        
        print(f"\n📂 Processing folder: {folder_name}")
        if not os.path.exists(checkpoint_path):
            print(f"  ❌ last.pt not found. Skipping.")
            continue
            
        # 1. Calculate Sparsity
        sparsity = get_checkpoint_sparsity(checkpoint_path)
        if sparsity is None:
            print(f"  ❌ Failed to calculate sparsity. Skipping.")
            continue
            
        # Determine the target sparsity (round to standard target sparsities: 80, 90, 95, 96, 97, 98, 99)
        target_sparsities = [0.8, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
        target_sparsity = min(target_sparsities, key=lambda x: abs(x - sparsity))
        
        print(f"  💡 Computed Sparsity: {sparsity * 100:.2f}% | Standard Target: {target_sparsity * 100:.1f}%")
        
        # 2. Run evaluate_model.py
        eval_env = os.environ.copy()
        eval_env["ENCODER_PATH"] = checkpoint_path
        eval_env["METHOD"] = "hebbian"
        eval_env["TARGET_DATASET"] = "cifar100"
        eval_env["NUM_EPOCHS"] = "50"
        eval_env["TARGET_SPARSITY"] = str(target_sparsity)
        
        # Set a dummy seed for evaluation run (it uses this seed internally for linear probing)
        eval_env["RUN_SEED"] = "42"
        
        print(f"  🚀 Running evaluate_model.py...")
        process = subprocess.Popen(
            ["python", "-u", "evaluate_model.py"],
            env=eval_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=MAIN_DIR,
            text=True
        )
        
        knn_acc = 0.0
        linear_acc = 0.0
        for line in process.stdout:
            print(f"    {line.strip()}")
            
            # Parse KNN accuracy: "KNN Accuracy: 37.14%" or similar
            m_knn = re.search(r"KNN Accuracy:\s*([0-9.]+)", line, re.IGNORECASE)
            if m_knn:
                knn_acc = float(m_knn.group(1)) / 100.0
                
            # Parse Linear Probing accuracy from output block: "Linear Probing: 52.83%"
            m_linear = re.search(r"Linear Probing:\s*([0-9.]+)", line, re.IGNORECASE)
            if m_linear:
                linear_acc = float(m_linear.group(1)) / 100.0
                
        process.wait()
        
        print(f"  📊 Folder: {folder_name} | Target: {target_sparsity * 100:.1f}% | KNN: {knn_acc * 100:.2f}% | Linear: {linear_acc * 100:.2f}%")
        
        # Save to direct_eval_summary.csv
        with open(SUMMARY_CSV, "a") as f:
            f.write(f"{folder_name},{target_sparsity:.4f},{sparsity:.4f},{knn_acc:.4f},{linear_acc:.4f}\n")
            
    print("\n============================================================")
    print(f"🎉 Direct Evaluation completed! Results saved to: {SUMMARY_CSV}")
    print("============================================================")

if __name__ == "__main__":
    main()
