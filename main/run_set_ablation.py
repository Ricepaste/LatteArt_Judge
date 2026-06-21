# main/run_set_ablation.py
import os
import subprocess
import time
import glob
import sys

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

def find_specific_run(runs_dir, target_sparsity, run_seed):
    sparsity_percent = int(float(target_sparsity) * 100)
    pattern = f"SET_SSL_s{sparsity_percent}_seed{run_seed}_*"
    folders = glob.glob(os.path.join(runs_dir, pattern))
    if not folders:
        return None
    folders.sort(key=os.path.getmtime)
    return folders[-1]

def main():
    runs_dir = os.path.join(MAIN_DIR, "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    log_dir = os.path.join(MAIN_DIR, "ablation_logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Read variables from environment, default to 80% sparsity
    run_seed = os.environ.get("RUN_SEED", "42")
    target_sparsity = os.environ.get("TARGET_SPARSITY", "0.8")
    target_dataset = os.environ.get("TARGET_DATASET", "cifar100")
    num_epochs = os.environ.get("NUM_EPOCHS", "400")
    
    log_file = os.path.join(log_dir, f"set_c100_{int(float(target_sparsity)*100)}_seed{run_seed}.log")
    
    # Environment variables for pretraining (we enable ABLATION_RANDOM_GROWTH and enforce original SET behavior)
    env = os.environ.copy()
    env["ABLATION_RANDOM_GROWTH"] = "1"
    env["USE_ERK"] = "False"
    env["PROTECT_HIGHWAY"] = "False"
    env["TARGET_DATASET"] = target_dataset
    env["TARGET_SPARSITY"] = target_sparsity
    env["NUM_EPOCHS"] = num_epochs
    env["RUN_SEED"] = run_seed
    
    print("=" * 60)
    print(f"🚀 Starting Standardized SET (Random Growth) Pretraining on {target_dataset.upper()}")
    print(f"Sparsity: {float(target_sparsity)*100:.1f}% | Epochs: {num_epochs} | Seed: {run_seed}")
    print(f"Log will be saved to: {log_file}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            f.write(f"=== PRETRAINING STAGE (STANDARDIZED SET / RANDOM GROWTH @ {float(target_sparsity)*100:.1f}% SPARSITY) ===\n")
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

    # 2. Find the saved checkpoint (specific to this run's sparsity and seed)
    latest_run = find_specific_run(runs_dir, target_sparsity, run_seed)
    if not latest_run:
        print(f"❌ Could not locate the run folder for sparsity {target_sparsity} and seed {run_seed}.")
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
    eval_env["METHOD"] = "hebbian" # Must be "hebbian" because the model class is Hebbian_SimSiam
    eval_env["USE_ERK"] = "False"
    eval_env["PROTECT_HIGHWAY"] = "False"
    eval_env["TARGET_DATASET"] = target_dataset
    eval_env["NUM_EPOCHS"] = "50"
    eval_env["TARGET_SPARSITY"] = target_sparsity
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
