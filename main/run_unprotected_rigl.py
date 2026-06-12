# main/run_unprotected_rigl.py
import os
import subprocess
import time
import glob

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

def find_latest_rigl_run(runs_dir):
    # find both shuffleNet_v05_SimSiam_* and shuffleNet_v05_SimSiam_
    folders = glob.glob(os.path.join(runs_dir, "shuffleNet_v05_SimSiam_*"))
    base_folder = os.path.join(runs_dir, "shuffleNet_v05_SimSiam_")
    if os.path.exists(base_folder):
        folders.append(base_folder)
        
    if not folders:
        return None
    folders.sort(key=os.path.getmtime)
    return folders[-1]

def main():
    runs_dir = os.path.join(MAIN_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    log_dir = os.path.join(MAIN_DIR, "ablation_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "unprotected_rigl_c100_99_seed3407.log")
    
    # Environment variables for pretraining
    env = os.environ.copy()
    env["RIGL_NO_PROTECT_FIRST_LAYER"] = "1"
    env["TARGET_DATASET"] = "cifar100"
    env["TARGET_SPARSITY"] = "0.99"
    env["NUM_EPOCHS"] = "400"
    env["RUN_SEED"] = "3407"
    
    print("=" * 60)
    print("🚀 Starting RigL Pretraining (Unprotected First Layer) on CIFAR-100")
    print(f"Sparsity: 99% | Epochs: 400 | Seed: 3407")
    print(f"Log will be saved to: {log_file}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            f.write("=== PRETRAINING STAGE ===\n")
            process = subprocess.Popen(
                ["python", "-u", "SimSiam.py"],
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=MAIN_DIR
            )
            process.wait()
            
        if process.returncode != 0:
            print(f"❌ Pretraining failed with return code {process.returncode}. Check: {log_file}")
            return
            
        duration = (time.time() - start_time) / 60
        print(f"✅ Pretraining finished successfully in {duration:.1f} minutes.")
    except Exception as e:
        print(f"❌ Error during pretraining: {e}")
        return

    # 2. Find the saved checkpoint
    latest_run = find_latest_rigl_run(runs_dir)
    if not latest_run:
        print("❌ Could not locate the latest RigL run folder.")
        return
        
    encoder_path = os.path.join(latest_run, "last.pt")
    if not os.path.exists(encoder_path):
        print(f"❌ Checkpoint file not found: {encoder_path}")
        return

    # 3. Standardized Evaluation Stage
    print("\n" + "=" * 60)
    print(f"🚀 Starting Standardized Evaluation on {encoder_path}")
    print("=" * 60)
    
    eval_env = os.environ.copy()
    eval_env["ENCODER_PATH"] = encoder_path
    eval_env["METHOD"] = "rigl"
    eval_env["TARGET_DATASET"] = "cifar100"
    eval_env["NUM_EPOCHS"] = "50"
    eval_env["TARGET_SPARSITY"] = "0.99"
    
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
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")

if __name__ == "__main__":
    main()
