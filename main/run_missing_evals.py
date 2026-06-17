# main/run_missing_evals.py
import os
import glob
import re
import subprocess
import sys
import time

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(MAIN_DIR, "runs")
LOG_DIR = os.path.join(MAIN_DIR, "ablation_logs")

def get_file_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0

def parse_log_filename(filename):
    # Match set_c100_{sparsity}_seed{seed}.log
    match = re.match(r"set_c100_(\d+)_seed(\d+)\.log", filename)
    if match:
        sparsity_percent = int(match.group(1))
        seed = match.group(2)
        # Convert sparsity_percent (e.g. 80, 95, 99) to float string (e.g. 0.8, 0.95, 0.99)
        if sparsity_percent == 80:
            sparsity = "0.8"
        else:
            sparsity = f"0.{sparsity_percent}"
        return sparsity, seed
    return None, None

def is_eval_missing_or_failed(log_path):
    if not os.path.exists(log_path):
        return True
    
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()
    
    # Check if we finished the linear probing accuracy block successfully
    has_started_eval = "=== STANDARDIZED EVALUATION STAGE ===" in content
    has_final_results = "Final Linear Probing Accuracy" in content or "FINAL RESULTS for" in content
    has_traceback = "Traceback" in content or "NameError" in content
    
    return (not has_started_eval) or (not has_final_results) or has_traceback

def clean_traceback_from_log(log_path):
    """
    If the log ended with a Python traceback or an incomplete evaluation attempt,
    remove those lines so the log stays clean before we append the new evaluation output.
    """
    if not os.path.exists(log_path):
        return
    
    with open(log_path, "r", errors="ignore") as f:
        lines = f.readlines()
        
    truncate_idx = None
    for idx, line in enumerate(lines):
        if "Traceback (most recent call last):" in line or "=== STANDARDIZED EVALUATION STAGE ===" in line:
            truncate_idx = idx
            break
            
    if truncate_idx is not None:
        print(f"🧹 Truncating failed/incomplete evaluation output from log: {os.path.basename(log_path)}")
        with open(log_path, "w") as f:
            f.writelines(lines[:truncate_idx])

def main():
    print("============================================================")
    print("🔍 Scanning for completed pre-trainings missing evaluation...")
    print("============================================================")
    
    # 1. Find all runs folders containing last.pt
    run_folders = glob.glob(os.path.join(RUNS_DIR, "Hebbian_SSL_*"))
    checkpoints = []
    for folder in run_folders:
        last_pt = os.path.join(folder, "last.pt")
        if os.path.exists(last_pt):
            checkpoints.append({
                "folder": folder,
                "checkpoint": last_pt,
                "mtime": get_file_mtime(last_pt)
            })
            
    if not checkpoints:
        print("❌ No completed ResNet checkpoints found in main/runs/.")
        sys.exit(1)
        
    print(f"Found {len(checkpoints)} saved checkpoints.")
    
    # 2. Find all set_c100_*.log files in log directory
    log_files = glob.glob(os.path.join(LOG_DIR, "set_c100_*.log"))
    
    # 根據修改時間由舊到新排序 (Oldest to Newest)
    log_files.sort(key=get_file_mtime)
    
    evaluated_count = 0
    for log_path in log_files:
        log_name = os.path.basename(log_path)
        log_mtime = get_file_mtime(log_path)
        
        # 1. 檢查是否已經有合法的評估結果
        if not is_eval_missing_or_failed(log_path):
            print(f"✅ {log_name} already has valid evaluation results. Skipping.")
            continue
            
        # 2. 檢查是否仍在訓練中 (若日誌在最近 5 分鐘內有寫入，表示該 Job 仍在運作，跳過)
        time_since_modified = time.time() - log_mtime
        if time_since_modified < 300: # 5 分鐘
            print(f"⏳ {log_name} was modified recently ({time_since_modified:.1f}s ago). It is likely still training. Skipping.")
            continue
            
        sparsity, seed = parse_log_filename(log_name)
        if not sparsity:
            continue
            
        print(f"\n⏳ Found log missing evaluation: {log_name} (Sparsity: {sparsity}, Seed: {seed})")
        
        # Match this log file to the checkpoint with the closest modification time
        # (Since they were written by the same job, the checkpoint save and the log file closing are near-simultaneous)
        best_checkpoint = None
        min_diff = float('inf')
        
        for cp in checkpoints:
            diff = abs(cp["mtime"] - log_mtime)
            if diff < min_diff:
                min_diff = diff
                best_checkpoint = cp
                
        # We accept a match if the mtime difference is within 10 minutes
        if best_checkpoint and min_diff < 600:
            print(f"🎯 Matched to checkpoint: {os.path.basename(best_checkpoint['folder'])} (Time difference: {min_diff:.1f}s)")
            
            # Clean up the traceback from the log file first
            clean_traceback_from_log(log_path)
            
            # Run evaluate_model.py
            eval_env = os.environ.copy()
            eval_env["ENCODER_PATH"] = best_checkpoint["checkpoint"]
            eval_env["METHOD"] = "hebbian"
            eval_env["TARGET_DATASET"] = "cifar100"
            eval_env["NUM_EPOCHS"] = "50"
            eval_env["TARGET_SPARSITY"] = sparsity
            eval_env["RUN_SEED"] = seed
            
            print(f"🚀 Running evaluate_model.py for checkpoint...")
            try:
                with open(log_path, "a") as f:
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
                    print(f"✅ Finished evaluating {log_name} successfully!")
                    evaluated_count += 1
                else:
                    print(f"❌ Evaluation process failed for {log_name} with code {process.returncode}")
            except Exception as e:
                print(f"❌ Error running evaluation: {e}")
        else:
            print(f"⚠️ Could not find a matching checkpoint folder for log {log_name} (closest difference: {min_diff:.1f}s)")
            
    print("\n" + "=" * 60)
    print(f"🎉 Process completed. Successfully evaluated {evaluated_count} missing runs.")
    print("============================================================")

if __name__ == "__main__":
    main()
