import os
import subprocess
import time

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 🌍 Transfer Learning Configuration (跨資料集遷移測試)
# =====================================================================
MODELS_TO_TEST = [
    {
        "name": "Hebbian_99_C100_Source",
        "method": "hebbian",
        "encoder_path": "/app/main/runs/Hebbian_SSL_20260410-190945/last.pt" 
    },
    {
        "name": "RigL_99_C100_Source",
        "method": "rigl",
        "encoder_path": "/app/main/runs/shuffleNet_v05_SimSiam__4/last.pt" 
    }
]

# 我們要遷移到的目標資料集 (例如從 CIFAR-100 訓練，遷移到 CIFAR-10 評估)
TARGET_DATASETS = ["cifar10"]

def run_transfer_evaluation(model_info, target_ds):
    print(f"\n{'='*60}")
    print(f"🚀 Starting Transfer Test: {model_info['name']} -> {target_ds.upper()}")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = target_ds
    run_env["NUM_EPOCHS"] = "50" # 遷移學習通常只需要較短的 Linear Probing 即可收斂
    
    # 強制關閉先前的輸入雜訊
    run_env["INPUT_NOISE_STD"] = "0.0"
    
    script = "lesion_evaluation.py" # 沿用舊檔名，但內容已更新為遷移學習
    cmd = ["python", "-u", script]
    
    # 將 log 存入專屬資料夾
    transfer_log_dir = os.path.join(MAIN_DIR, "transfer_logs")
    os.makedirs(transfer_log_dir, exist_ok=True)
    log_file = os.path.join(transfer_log_dir, f"transfer_{model_info['name']}_to_{target_ds}.log")
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            
            while process.poll() is None:
                time.sleep(1)
                
            if process.returncode == 0:
                print(f"✅ Finished! Log saved to: {log_file}")
            else:
                print(f"❌ Failed with return code {process.returncode}. Check log: {log_file}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    print("🌟 Starting Automated Transfer Learning Generalization Tests 🌟\n")
    
    for model in MODELS_TO_TEST:
        if not os.path.exists(os.path.join(MAIN_DIR, model["encoder_path"])) and not os.path.exists(model["encoder_path"]):
            print(f"⚠️ 找不到權重檔案: {model['encoder_path']}，跳過 {model['name']}。")
            continue
            
        for ds in TARGET_DATASETS:
            run_transfer_evaluation(model, ds)
            
    print("\n🎉 All Transfer Tests Finished! 請檢查 main/transfer_results_*.txt 裡的數據！")
