# main/run_positive_hebb_80.py
import os
import subprocess
import time
import glob
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

def find_latest_hebbian_run(runs_dir):
    # Hebbian runs are named "Hebbian_SSL_*"
    folders = glob.glob(os.path.join(runs_dir, "Hebbian_SSL_*"))
    if not folders:
        return None
    folders.sort(key=os.path.getmtime)
    return folders[-1]

def main():
    runs_dir = os.path.join(MAIN_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    log_dir = os.path.join(MAIN_DIR, "ablation_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Check if run_seed is passed via env, default to 42
    run_seed = os.environ.get("RUN_SEED", "42")
    num_epochs = os.environ.get("NUM_EPOCHS", "400")
    
    log_file = os.path.join(log_dir, f"positive_hebb_c100_80_seed{run_seed}.log")
    
    # Environment variables for pretraining
    env = os.environ.copy()
    env["ABLATION_POSITIVE_HEBB_ONLY"] = "1"
    env["TARGET_DATASET"] = "cifar100"
    env["TARGET_SPARSITY"] = "0.8"
    env["NUM_EPOCHS"] = num_epochs
    env["RUN_SEED"] = run_seed
    
    print("=" * 60)
    print("🚀 Starting Positive Hebbian Ablation Pretraining on CIFAR-100 (Sparsity: 80%)")
    print(f"Sparsity: 80% | Epochs: {num_epochs} | Seed: {run_seed}")
    print(f"Log will be saved to: {log_file}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            f.write("=== PRETRAINING STAGE (POSITIVE HEBBIAN ONLY @ 80% SPARSITY) ===\n")
            process = subprocess.Popen(
                ["python", "-u", "Hebbian.py"],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=MAIN_DIR
            )
            process.wait()
            
        if process.returncode != 0:
            print(f"❌ Pretraining failed with return code {process.returncode}. Check: {log_file}")
            sys.exit(process.returncode)
            
        duration = (time.time() - start_time) / 60
        print(f"✅ Pretraining finished successfully in {duration:.1f} minutes.")
    except Exception as e:
        print(f"❌ Error during pretraining: {e}")
        sys.exit(1)

    # 2. Find the saved checkpoint
    latest_run = find_latest_hebbian_run(runs_dir)
    if not latest_run:
        print("❌ Could not locate the latest Hebbian run folder.")
        sys.exit(1)
        
    encoder_path = os.path.join(latest_run, "last.pt")
    if not os.path.exists(encoder_path):
        print(f"❌ Checkpoint file not found: {encoder_path}")
        sys.exit(1)

    # 3. Standardized Evaluation Stage
    print("\n" + "=" * 60)
    print(f"🚀 Starting Standardized Evaluation on {encoder_path}")
    print("=" * 60)
    
    eval_env = os.environ.copy()
    eval_env["ENCODER_PATH"] = encoder_path
    eval_env["METHOD"] = "hebbian"
    eval_env["TARGET_DATASET"] = "cifar100"
    eval_env["NUM_EPOCHS"] = "50"
    eval_env["TARGET_SPARSITY"] = "0.8"
    eval_env["RUN_SEED"] = run_seed
    
    try:
        with open(log_file, "a") as f:
            f.write("\n\n" + "=" * 50 + "\n")
            f.write("=== STANDARDIZED EVALUATION STAGE ===\n")
            f.write("=" * 50 + "\n")
            process = subprocess.Popen(
                ["python", "-u", "evaluate_model.py"],
                env=eval_env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=MAIN_DIR
            )
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ Evaluation completed successfully! Full log: {log_file}")
        else:
            print(f"❌ Evaluation failed with code {process.returncode}. Check log: {log_file}")
            sys.exit(process.returncode)
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
