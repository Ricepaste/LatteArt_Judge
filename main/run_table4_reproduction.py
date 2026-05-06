import os
import subprocess
import time
import pandas as pd
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 📋 Table 4 Reproduction List (路線一：全面升級評估標準)
# =====================================================================
MODELS_TO_TEST = [
    # --- 伺服器一 (RTX 4090) ---
    {"name": "RigL_80", "method": "rigl", "path": "/app/main/runs/shuffleNet_v05_SimSiam__2/last.pt"},
    {"name": "RigL_90", "method": "rigl", "path": "/app/main/runs/shuffleNet_v05_SimSiam__1/last.pt"},
    {"name": "RigL_95", "method": "rigl", "path": "/app/main/runs/shuffleNet_v05_SimSiam_/last.pt"},
    
    {"name": "Hebbian_80", "method": "hebbian", "path": "/app/main/runs/Hebbian_SSL_20260415-010443/last.pt"},
    {"name": "Hebbian_90", "method": "hebbian", "path": "/app/main/runs/Hebbian_SSL_20260416-091440/last.pt"},
    {"name": "Hebbian_95", "method": "hebbian", "path": "/app/main/runs/Hebbian_SSL_20260417-172310/last.pt"},
    
    {"name": "Hebbian_99_seed42", "method": "hebbian", "path": "/app/main/runs/Hebbian_SSL_20260410-190945/last.pt"},
    {"name": "Hebbian_99_seed3407", "method": "hebbian", "path": "/app/main/runs/Hebbian_SSL_20260412-050458/last.pt"},
    {"name": "Hebbian_99_seed114514", "method": "hebbian", "path": "/app/main/runs/Hebbian_SSL_20260413-163009/last.pt"},

    # --- 伺服器二 (RTX 3090) ---
    {"name": "RigL_99_seed42", "method": "rigl", "path": "/app/main/runs/shuffleNet_v05_SimSiam__4/last.pt"},
    {"name": "RigL_99_seed3407", "method": "rigl", "path": "/app/main/runs/shuffleNet_v05_SimSiam__5/last.pt"},
    {"name": "RigL_99_seed114514", "method": "rigl", "path": "/app/main/runs/shuffleNet_v05_SimSiam__7/last.pt"},
]

# 這次重跑是針對 CIFAR-100 (主實驗資料集)
TARGET_DATASET = "cifar100"
MASTER_CSV = os.path.join(MAIN_DIR, "table4_reproduced_summary.csv")

def run_evaluation(model_info):
    print(f"\n>>>> [REPRODUCING] {model_info['name']} | Method: {model_info['method']}")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = TARGET_DATASET
    run_env["NUM_EPOCHS"] = "50" 
    run_env["EVAL_FRACTION"] = "1.0"
    
    # 確保 CSV 寫入到我們指定的新檔案
    run_env["SUMMARY_FILE_OVERRIDE"] = MASTER_CSV
    
    script = "evaluate_model.py"
    cmd = ["python", "-u", script]
    
    log_dir = os.path.join(MAIN_DIR, "reproduction_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{model_info['name']}.log")
    
    try:
        with open(log_path, "w") as f:
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ Success. Results added to {MASTER_CSV}")
        else:
            print(f"❌ Failed. Check: {log_path}")
                
    except KeyboardInterrupt:
        process.terminate()
        return False
    return True

if __name__ == "__main__":
    # 清除舊的 CSV 重新開始
    if os.path.exists(MASTER_CSV): os.remove(MASTER_CSV)
    
    print("\n" + "="*80)
    print("🌟 TABLE 4 REPRODUCTION: UPGRADING EVALUATION STANDARDS")
    print("="*80)
    
    start_time = time.time()
    for model in MODELS_TO_TEST:
        if not os.path.exists(model["path"]):
            print(f"⚠️ Warning: File not found: {model['path']}")
            continue
        if not run_evaluation(model): break
    
    print(f"\n🎉 Reproduction Finished! Total Time: {(time.time()-start_time)/60:.1f} mins.")
    
    if os.path.exists(MASTER_CSV):
        print("\n" + "📊 REPRODUCED TABLE 4 DATA ".center(80, "="))
        df = pd.read_csv(MASTER_CSV)
        # 加入一點簡單的排序方便閱讀
        df['Sparsity'] = df['Source_Model'].apply(lambda x: x.split('_')[1] if '_' in x else '99')
        print(df.to_string(index=False))
        print("="*80)
