import os
import subprocess
import time

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 🏥 Lesion Test Configuration (請填入你跑好的 best.pt 絕對或相對路徑)
# =====================================================================
MODELS_TO_TEST = [
    {
        "name": "Hebbian_99_CIFAR100",
        "method": "hebbian",
        "dataset": "cifar100",
        "encoder_path": "runs/你的hebbian_seed42的資料夾/best.pt"  # 🔴 請替換成真實路徑
    },
    {
        "name": "RigL_99_CIFAR100",
        "method": "rigl",
        "dataset": "cifar100",
        "encoder_path": "runs/你的rigl_seed42的資料夾/best.pt"     # 🔴 請替換成真實路徑
    }
]

# 我們要測試的破壞比例：從 5% 到 30%
LESION_RATIOS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def run_lesion_evaluation(model_info, ratio):
    print(f"\n{'='*50}")
    print(f"🚀 Starting Lesion Test: {model_info['name']} @ {ratio*100}% Damage")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = model_info["dataset"]
    run_env["LESION_RATIO"] = str(ratio)
    
    # 強制關閉先前的輸入雜訊，確保公平評估
    run_env["INPUT_NOISE_STD"] = "0.0"
    
    script = "lesion_evaluation.py"
    cmd = ["python", "-u", script]
    
    # 將 log 存入專屬資料夾
    lesion_log_dir = os.path.join(MAIN_DIR, "lesion_logs")
    os.makedirs(lesion_log_dir, exist_ok=True)
    log_file = os.path.join(lesion_log_dir, f"lesion_{model_info['name']}_{int(ratio*100)}percent.log")
    
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
    print("🌟 Starting Automated Lesion Tolerance Tests 🌟\n")
    
    for model in MODELS_TO_TEST:
        # 檢查路徑是否存在，避免跑錯
        if not os.path.exists(os.path.join(MAIN_DIR, model["encoder_path"])):
            print(f"⚠️ 找不到權重檔案: {model['encoder_path']}，跳過 {model['name']} 的測試。請務必填寫正確路徑！")
            continue
            
        for ratio in LESION_RATIOS:
            run_lesion_evaluation(model, ratio)
            
    print("\n🎉 All Lesion Tests Finished! 所有的結果已經統整到 main/lesion_results_*.txt 裡了！")
