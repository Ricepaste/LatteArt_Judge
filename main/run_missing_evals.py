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
        # Convert sparsity_percent (e.g. 80, 90, 95) to float string (e.g. 0.8, 0.9, 0.95)
        sparsity = str(sparsity_percent / 100)
        return sparsity, seed
    return None, None

def find_checkpoint_path_in_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()
    
    # Match Hebbian_SSL_YYYYMMDD-HHMMSS anywhere in the log
    match = re.search(r"Hebbian_SSL_\d{8}-\d{6}", content)
    if match:
        return os.path.join(RUNS_DIR, match.group(0), "last.pt")
    return None

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

def is_pretraining_finished(log_path):
    if not os.path.exists(log_path):
        return False
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()
    return "Training Finished." in content or "Epoch 400" in content

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
        
        missing = is_eval_missing_or_failed(log_path)
        finished = is_pretraining_finished(log_path)
        
        print(f"\n🔍 Scanning log: {log_name}")
        print(f"  - Evaluation missing/failed: {missing}")
        print(f"  - Pre-training finished: {finished}")
        
        # 1. 檢查是否已經有合法的評估結果
        if not missing:
            print(f"  ✅ Already has valid evaluation results. Skipping.")
            continue
            
        # 2. 檢查預訓練是否已經完成 (如果沒完成，表示仍在訓練，跳過)
        if not finished:
            print(f"  ⏳ Still pre-training. Skipping.")
            continue
            
        sparsity, seed = parse_log_filename(log_name)
        if not sparsity:
            print(f"  ⚠️ Could not parse sparsity/seed from filename.")
            continue
            
        print(f"  💡 Targets: Sparsity={sparsity}, Seed={seed}")
        
        # 1. 優先嘗試從日誌內容中尋找之前失敗/中斷的權重路徑
        checkpoint_path = find_checkpoint_path_in_log(log_path)
        best_checkpoint = None
        min_diff = float('inf')
        
        if checkpoint_path:
            exists = os.path.exists(checkpoint_path)
            print(f"  🔍 Regex found checkpoint path: {checkpoint_path} (exists={exists})")
            if exists:
                best_checkpoint = {
                    "checkpoint": checkpoint_path,
                    "folder": os.path.dirname(checkpoint_path)
                }
                print(f"  🎯 Matched via log content: {os.path.basename(best_checkpoint['folder'])}")
            else:
                print(f"  ⚠️ Checkpoint path in log does not exist on disk.")
        
        if not best_checkpoint:
            # 2. 備用方案：根據修改時間進行最接近配對 (容許落差 24 小時)
            print(f"  ⏳ Attempting fallback to modification time matching...")
            min_diff = float('inf')
            matched_cp = None
            for cp in checkpoints:
                diff = abs(cp["mtime"] - log_mtime)
                if diff < min_diff:
                    min_diff = diff
                    matched_cp = cp
                    
            if matched_cp and min_diff < 86400: # 24 小時
                best_checkpoint = matched_cp
                print(f"  🎯 Matched via modification time: {os.path.basename(best_checkpoint['folder'])} (Time difference: {min_diff:.1f}s)")
            else:
                if matched_cp:
                    print(f"  ❌ Closest checkpoint time difference too large ({min_diff:.1f}s, limit 24h).")
                else:
                    print(f"  ❌ No checkpoints found in checkpoints list.")
                
        if best_checkpoint:
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
            
            print(f"  🚀 Running evaluate_model.py for checkpoint...")
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
                    print(f"  ✅ Finished evaluating {log_name} successfully!")
                    evaluated_count += 1
                else:
                    print(f"  ❌ Evaluation process failed for {log_name} with code {process.returncode}")
            except Exception as e:
                print(f"  ❌ Error running evaluation: {e}")
        else:
            print(f"  ❌ Could not find a matching checkpoint folder for log {log_name}")
            
    print("\n" + "=" * 60)
    print(f"🎉 Process completed. Successfully evaluated {evaluated_count} missing runs.")
    print("============================================================")

if __name__ == "__main__":
    main()
