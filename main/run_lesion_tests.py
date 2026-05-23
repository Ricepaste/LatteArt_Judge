import os
import subprocess
import time
import pandas as pd
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 🌍 Transfer & Semi-Supervised Configuration
# =====================================================================
MODELS_TO_TEST = [
    {
        "name": "Hebbian_99_C100",
        "method": "hebbian",
        "encoder_path": "/app/main/runs/Hebbian_SSL_20260410-190945/last.pt" 
    },
    # {
    #     "name": "RigL_99_C100",
    #     "method": "rigl",
    #     "encoder_path": "/app/main/runs/shuffleNet_v05_SimSiam__4/last.pt" 
    # }
]

# 測試目標：涵蓋同領域、跨領域、與特殊視覺領域 (Texture, Medical, Satellite)
# TARGET_DATASETS = ["svhn", "stl10", "eurosat"]
TARGET_DATASETS = ["cifar100", "cifar10", "svhn", "stl10", "eurosat", "dtd", "pcam"]

# 少樣本比例：100% (遷移能力)
EVAL_FRACTIONS = [1.0] 

def run_evaluation(model_info, target_ds, fraction):
    print(f"\n>>>> [RUNNING] {model_info['name']} on {target_ds.upper()} ({fraction*100}% labels)")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = target_ds
    run_env["NUM_EPOCHS"] = "50" 
    run_env["EVAL_FRACTION"] = str(fraction)
    run_env["INPUT_NOISE_STD"] = "0.0"
    
    script = "evaluate_model.py"
    cmd = ["python", "-u", script]
    
    # 建立時間戳記資料夾，避免 Log 混亂
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(MAIN_DIR, "eval_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{model_info['method']}_{target_ds}_{int(fraction*100)}pct.log"
    log_path = os.path.join(log_dir, log_filename)
    
    try:
        with open(log_path, "w") as f:
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ Success. Log: {log_path}")
        else:
            print(f"❌ Failed. Check: {log_path}")
                
    except KeyboardInterrupt:
        process.terminate()
        print("\n⚠️ Interrupted.")
        return False
    return True

if __name__ == "__main__":
    # 清除舊的 Master CSV (若你想重新開始收集)
    # if os.path.exists("transfer_summary_master.csv"): os.remove("transfer_summary_master.csv")
    
    print("\n" + "="*60)
    print("🚀 AUTOMATED EVALUATION SUITE STARTING")
    print("="*60)
    
    start_time = time.time()
    
    for model in MODELS_TO_TEST:
        if not os.path.exists(model["encoder_path"]):
            print(f"⚠️ Skip: {model['encoder_path']} not found.")
            continue
            
        for ds in TARGET_DATASETS:
            for frac in EVAL_FRACTIONS:
                if not run_evaluation(model, ds, frac):
                    break
    
    duration = (time.time() - start_time) / 60
    print(f"\n🎉 All tests finished in {duration:.1f} minutes.")
    
    # --- 讀取並列印總結表格 ---
    master_csv = os.path.join(MAIN_DIR, "transfer_summary_master.csv")
    if os.path.exists(master_csv):
        print("\n" + "📊 MASTER SUMMARY TABLE ".center(80, "="))
        df = pd.read_csv(master_csv)
        # 只顯示最後這一輪產生的結果（若 CSV 很大）
        print(df.tail(len(MODELS_TO_TEST) * len(TARGET_DATASETS) * len(EVAL_FRACTIONS)).to_string(index=False))
        print("="*80)
        print(f"Full data saved in: {master_csv}")
    else:
        print("\n❌ Error: No results collected in transfer_summary_master.csv")
