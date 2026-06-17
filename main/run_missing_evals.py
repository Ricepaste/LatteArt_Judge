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
DEBUG_LOG_PATH = os.path.join(LOG_DIR, "recovery_debug.log")

def get_file_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0

def log_print(msg):
    # Print to stdout with flush=True
    print(msg, flush=True)
    # Append to recovery_debug.log
    try:
        with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(msg + "\n")
    except Exception:
        pass

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

def parse_start_time_from_folder(folder_name):
    # Extract timestamp from Hebbian_SSL_YYYYMMDD-HHMMSS
    match = re.search(r"Hebbian_SSL_(\d{8})-(\d{6})", folder_name)
    if match:
        time_str = f"{match.group(1)}-{match.group(2)}"
        try:
            struct_time = time.strptime(time_str, "%Y%m%d-%H%M%S")
            return time.mktime(struct_time)
        except Exception:
            pass
    return None

def parse_pretraining_duration_from_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()
    
    # Match "Training Finished. Total Time: 2.15 hours."
    match = re.search(r"Training Finished\.\s*Total\s*Time:\s*([0-9.]+)\s*hours", content)
    if match:
        return float(match.group(1)) * 3600 # Convert to seconds
    return None

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
            return round(1.0 - (active_elements / total_elements), 4)
    except Exception:
        pass
    return None

def find_checkpoint_path_in_log(log_path):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", errors="ignore") as f:
        content = f.read()
    
    # 1. Look for Source Model Path:
    match = re.search(r"Source Model Path:.*?(Hebbian_SSL_\d{8}-\d{6})", content)
    if match:
        return os.path.join(RUNS_DIR, match.group(1), "last.pt")
        
    # 2. Look for Starting Standardized Evaluation on
    match = re.search(r"Starting Standardized Evaluation on.*?(Hebbian_SSL_\d{8}-\d{6})", content)
    if match:
        return os.path.join(RUNS_DIR, match.group(1), "last.pt")
        
    return None

def build_slurm_logs_mapping():
    project_root = os.path.dirname(MAIN_DIR)
    logs_dir = os.path.join(project_root, "logs")
    mapping = {} # (sparsity_str, seed_str) -> run_folder_name
    
    if not os.path.exists(logs_dir):
        return mapping
        
    out_files = glob.glob(os.path.join(logs_dir, "*.out"))
    # Sort by modification time (oldest to newest) so newer runs overwrite older ones in the dict
    out_files.sort(key=get_file_mtime)
    
    for fpath in out_files:
        try:
            with open(fpath, "r", errors="ignore") as f:
                content = f.read()
            
            sparsity = None
            seed = None
            
            m1 = re.search(r"Target Sparsity:\s*([0-9.]+)\s*\|\s*Seed:\s*(\d+)", content)
            if m1:
                sparsity = str(float(m1.group(1)))
                seed = m1.group(2)
            else:
                m2 = re.search(r"Sparsity:\s*([0-9.]+)%\s*\|\s*Epochs:\s*\d+\s*\|\s*Seed:\s*(\d+)", content)
                if m2:
                    sparsity = str(float(m2.group(1)) / 100)
                    seed = m2.group(2)
            
            m_folder = re.search(r"Hebbian_SSL_\d{8}-\d{6}", content)
            
            if sparsity and seed and m_folder:
                norm_sparsity = str(float(sparsity))
                mapping[(norm_sparsity, seed)] = m_folder.group(0)
        except Exception:
            pass
            
    return mapping

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
    
    if "Training Finished." in content:
        return True
        
    epochs = re.findall(r"Epoch (\d+)", content)
    if epochs:
        last_epoch = int(epochs[-1])
        match_total = re.search(r"Total Epochs:\s*(\d+)", content)
        if match_total:
            total_epochs = int(match_total.group(1))
            if last_epoch >= total_epochs:
                return True
        else:
            if last_epoch >= 400:
                return True
                
    return "Epoch 400" in content

def clean_traceback_from_log(log_path):
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
        log_print(f"🧹 Truncating failed/incomplete evaluation output from log: {os.path.basename(log_path)}")
        with open(log_path, "w") as f:
            f.writelines(lines[:truncate_idx])

def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Initialize recovery_debug.log
    try:
        with open(DEBUG_LOG_PATH, "w", encoding="utf-8") as f:
            f.write(f"=== Recovery Scan Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    except Exception as e:
        print(f"Error initializing debug log: {e}")

    log_print("============================================================")
    log_print("🔍 Scanning for completed pre-trainings missing evaluation...")
    log_print("============================================================")
    
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
        log_print("❌ No completed ResNet checkpoints found in main/runs/.")
        sys.exit(1)
        
    log_print(f"Found {len(checkpoints)} saved checkpoints on disk.")
    
    # Build Slurm logs mapping (Primary source of truth for matching seed/sparsity to checkpoints)
    log_print("\n📂 Building job mappings from Slurm log files (*.out)...")
    slurm_mapping = build_slurm_logs_mapping()
    log_print(f"Loaded {len(slurm_mapping)} mappings from Slurm logs.")
    for k, v in slurm_mapping.items():
        log_print(f"  - Sparsity {k[0]}, Seed {k[1]} -> {v}")
    
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
        
        log_print(f"\n🔍 Scanning log: {log_name}")
        log_print(f"  - Evaluation missing/failed: {missing}")
        log_print(f"  - Pre-training finished: {finished}")
        
        # 1. 檢查是否已經有合法的評估結果
        if not missing:
            log_print(f"  ✅ Already has valid evaluation results. Skipping.")
            continue
            
        # 2. 檢查預訓練是否已經完成 (如果沒完成，表示仍在訓練，跳過)
        if not finished:
            log_print(f"  ⏳ Still pre-training. Skipping.")
            continue
            
        sparsity, seed = parse_log_filename(log_name)
        if not sparsity:
            log_print(f"  ⚠️ Could not parse sparsity/seed from filename.")
            continue
            
        log_print(f"  💡 Targets: Sparsity={sparsity}, Seed={seed}")
        
        best_checkpoint = None
        checkpoint_source = None
        
        # Priority 1: Match via Slurm log files (the most accurate)
        slurm_folder = slurm_mapping.get((sparsity, seed))
        if slurm_folder:
            checkpoint_path = os.path.join(RUNS_DIR, slurm_folder, "last.pt")
            exists = os.path.exists(checkpoint_path)
            log_print(f"  🔍 Slurm logs matched folder: {slurm_folder} (exists={exists})")
            if exists:
                best_checkpoint = {
                    "checkpoint": checkpoint_path,
                    "folder": os.path.dirname(checkpoint_path)
                }
                checkpoint_source = "Slurm log mapping"
        
        # Priority 2: Try parsing checkpoint path from log content (filtered to specific keywords)
        if not best_checkpoint:
            checkpoint_path = find_checkpoint_path_in_log(log_path)
            if checkpoint_path:
                exists = os.path.exists(checkpoint_path)
                log_print(f"  🔍 Log content matched path: {checkpoint_path} (exists={exists})")
                if exists:
                    best_checkpoint = {
                        "checkpoint": checkpoint_path,
                        "folder": os.path.dirname(checkpoint_path)
                    }
                    checkpoint_source = "Log content"
                else:
                    log_print(f"  ⚠️ Checkpoint path in log does not exist on disk.")
        
        # Priority 3: Smart Sparsity & Duration Matching fallback (vital when Slurm logs are cleared)
        if not best_checkpoint:
            log_print(f"  ⏳ Attempting smart matching via Sparsity & Duration alignment...")
            log_duration = parse_pretraining_duration_from_log(log_path)
            if log_duration is not None:
                log_print(f"    - Parsed pre-training duration from log: {log_duration/3600:.2f} hours ({log_duration:.1f} seconds)")
            else:
                log_print(f"    - Could not parse duration from log (might have failed mid-pretraining).")
                
            candidates = []
            for cp in checkpoints:
                # 1. Sparsity check
                cp_sparsity = get_checkpoint_sparsity(cp["checkpoint"])
                if cp_sparsity is None:
                    continue
                try:
                    match_sparsity = abs(cp_sparsity - float(sparsity)) < 0.01
                except Exception:
                    match_sparsity = False
                if not match_sparsity:
                    continue
                    
                # 2. Duration check
                start_time = parse_start_time_from_folder(os.path.basename(cp["folder"]))
                if start_time and cp["mtime"] > start_time:
                    cp_duration = cp["mtime"] - start_time
                    cp["duration"] = cp_duration
                    if log_duration is not None:
                        diff = abs(cp_duration - log_duration)
                        cp["duration_diff"] = diff
                        # Max 5 minutes duration difference tolerance
                        if diff <= 300:
                            candidates.append(cp)
                    else:
                        candidates.append(cp)
                else:
                    if log_duration is None:
                        candidates.append(cp)
            
            if candidates:
                if log_duration is not None:
                    # Sort by duration difference
                    candidates.sort(key=lambda x: x.get("duration_diff", float('inf')))
                    best_checkpoint = candidates[0]
                    checkpoint_source = f"Smart duration alignment (diff: {best_checkpoint['duration_diff']:.1f}s)"
                    log_print(f"  🎯 Matched via Sparsity & Duration: {os.path.basename(best_checkpoint['folder'])}")
                else:
                    # Fallback to modification time matching only among filtered sparsity candidates
                    log_mtime = get_file_mtime(log_path)
                    candidates.sort(key=lambda x: abs(x["mtime"] - log_mtime))
                    best_checkpoint = candidates[0]
                    time_diff = abs(best_checkpoint["mtime"] - log_mtime)
                    checkpoint_source = f"Sparsity-filtered modification time (diff: {time_diff:.1f}s)"
                    log_print(f"  🎯 Matched via Sparsity & Modification Time: {os.path.basename(best_checkpoint['folder'])}")
        
        # Priority 4: Final global modification time fallback
        if not best_checkpoint:
            log_print(f"  ⏳ Fallback to global modification time matching...")
            min_diff = float('inf')
            matched_cp = None
            for cp in checkpoints:
                diff = abs(cp["mtime"] - log_mtime)
                if diff < min_diff:
                    min_diff = diff
                    matched_cp = cp
                    
            if matched_cp and min_diff < 86400: # 24 小時
                best_checkpoint = matched_cp
                checkpoint_source = f"Global modification time difference ({min_diff:.1f}s)"
                log_print(f"  🎯 Matched via global modification time: {os.path.basename(best_checkpoint['folder'])}")
            else:
                if matched_cp:
                    log_print(f"  ❌ Closest checkpoint time difference too large ({min_diff:.1f}s, limit 24h).")
                else:
                    log_print(f"  ❌ No checkpoints found in checkpoints list.")
                
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
            
            log_print(f"  🚀 Running evaluate_model.py (Source: {checkpoint_source}) for checkpoint...")
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
                    log_print(f"  ✅ Finished evaluating {log_name} successfully!")
                    evaluated_count += 1
                else:
                    log_print(f"  ❌ Evaluation process failed for {log_name} with code {process.returncode}")
            except Exception as e:
                log_print(f"  ❌ Error running evaluation: {e}")
        else:
            log_print(f"  ❌ Could not find a matching checkpoint folder for log {log_name}")
            
    log_print("\n" + "=" * 60)
    log_print(f"🎉 Process completed. Successfully evaluated {evaluated_count} missing runs.")
    log_print("============================================================")

if __name__ == "__main__":
    main()
