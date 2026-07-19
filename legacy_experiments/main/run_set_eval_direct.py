# main/run_set_eval_direct.py
import os
import re
import subprocess
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(MAIN_DIR, "runs")
LOG_DIR = os.path.join(MAIN_DIR, "ablation_logs")
SUMMARY_CSV = os.path.join(LOG_DIR, "set_direct_eval_summary.csv")

FOLDERS = [
    "SET_SSL_s70_seed15303_in100_20260630-170718",
    "SET_SSL_s70_seed24564_in100_20260702-164939",
    "SET_SSL_s90_seed19551_in100_20260702-165023",
    "SET_SSL_s90_seed2373_in100_20260703-164103",
    "SET_SSL_s99_seed18942_in100_20260703-164125"
]

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    print("============================================================")
    print("🚀 SET Direct KNN & Linear Probing Evaluation Started")
    print("============================================================")
    
    # Initialize Summary CSV
    if not os.path.exists(SUMMARY_CSV):
        with open(SUMMARY_CSV, "w") as f:
            f.write("Run_Folder,Target_Sparsity,KNN_Acc,Linear_Acc\n")
            
    for folder_name in FOLDERS:
        folder_path = os.path.join(RUNS_DIR, folder_name)
        checkpoint_path = os.path.join(folder_path, "last.pt")
        
        print(f"\n📂 Processing folder: {folder_name}")
        if not os.path.exists(checkpoint_path):
            print(f"  ❌ last.pt not found at {checkpoint_path}. Skipping.")
            continue
            
        # Parse target sparsity from folder name (e.g., _s70_ -> 0.7)
        match = re.search(r"_s(\d+)_", folder_name)
        if match:
            target_sparsity = float(match.group(1)) / 100.0
        else:
            print(f"  ❌ Could not parse sparsity from folder name. Skipping.")
            continue
            
        # Parse seed from folder name (e.g., _seed15303_ -> 15303)
        match_seed = re.search(r"_seed(\d+)_", folder_name)
        seed = match_seed.group(1) if match_seed else "42"
        
        print(f"  💡 Parsed Target Sparsity: {target_sparsity * 100:.1f}% | Seed: {seed}")
        
        # Run evaluate_model.py
        eval_env = os.environ.copy()
        eval_env["ENCODER_PATH"] = checkpoint_path
        eval_env["METHOD"] = "set"
        eval_env["TARGET_DATASET"] = "imagenet100"
        eval_env["NUM_EPOCHS"] = "50"
        eval_env["TARGET_SPARSITY"] = str(target_sparsity)
        eval_env["USE_ERK"] = "False"
        eval_env["PROTECT_HIGHWAY"] = "False"
        eval_env["RUN_SEED"] = seed
        
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
            
            m_knn = re.search(r"KNN Accuracy:\s*([0-9.]+)", line, re.IGNORECASE)
            if m_knn:
                knn_acc = float(m_knn.group(1)) / 100.0
                
            m_linear = re.search(r"Linear Probing:\s*([0-9.]+)", line, re.IGNORECASE)
            if m_linear:
                linear_acc = float(m_linear.group(1)) / 100.0
                
        process.wait()
        
        print(f"  📊 Folder: {folder_name} | KNN: {knn_acc * 100:.2f}% | Linear: {linear_acc * 100:.2f}%")
        
        # Save to Summary CSV
        with open(SUMMARY_CSV, "a") as f:
            f.write(f"{folder_name},{target_sparsity:.4f},{knn_acc:.4f},{linear_acc:.4f}\n")
            
    print("\n============================================================")
    print(f"🎉 SET Direct Evaluation completed! Results saved to: {SUMMARY_CSV}")
    print("============================================================")

if __name__ == "__main__":
    main()
