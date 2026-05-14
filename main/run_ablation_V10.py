import os
import subprocess
import time
import glob
import sys

# 取得 main/ 目錄的絕對路徑，確保所有的 relative paths 都不受執行位置的影響
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Ablation Tests Configuration ---
# 1表示啟用, 0表示關閉

# --- 🚨 THESIS EMERGENCY PRIORITY MODE 🚨 ---
# 既然 1 個實驗要 2 天，全面 3-Seed 驗證是不可能的。我們必須把算力集中在「口試委員最會攻擊的地方」！

# 💀 Priority 1: 捍衛主戰場 (已完成)
GENERALIZATION = []

# ⚔️ Priority 2: 點出衰減交叉點 (瞄準 95% ~ 99% 的超車區間)
SPARSITY_CURVE = [
    {"name": "Sparsity_V8_96", "dataset": "cifar100", "script": "Hebbian.py", "sparsity": 0.96, "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}},
    {"name": "Sparsity_RigL_96", "dataset": "cifar100", "script": "SimSiam.py", "sparsity": 0.96, "env": {}},
    
    {"name": "Sparsity_V8_97", "dataset": "cifar100", "script": "Hebbian.py", "sparsity": 0.97, "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}},
    {"name": "Sparsity_RigL_97", "dataset": "cifar100", "script": "SimSiam.py", "sparsity": 0.97, "env": {}},
    
    {"name": "Sparsity_V8_98", "dataset": "cifar100", "script": "Hebbian.py", "sparsity": 0.98, "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}},
    {"name": "Sparsity_RigL_98", "dataset": "cifar100", "script": "SimSiam.py", "sparsity": 0.98, "env": {}},
]

# 🧪 Priority 4: Noise Robustness Challenge (CIFAR-100 @ 99% + 10% Input Noise)
# 用來模擬「環境惡化」時，誰的拓樸生長更穩健。
# 我們對 Input 注入 std=0.1 的高斯雜訊 (約 = 10% 數據污染)。
NOISE_ROBUSTNESS = [
    {"name": "Robust_V8_Full_C100_Noise0.1", "dataset": "cifar100", "script": "Hebbian.py", "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1", "INPUT_NOISE_STD": "0.1"}},
    {"name": "Robust_RigL_C100_Noise0.1", "dataset": "cifar100", "script": "SimSiam.py", "env": {"INPUT_NOISE_STD": "0.1"}}
]


# 🛡️ Priority 3: 舊資料防禦 (Ablation Tests)
ABLATIONS = []


def run_experiment(exp_name, env_vars, dataset="cifar10", script="Hebbian.py", sparsity=0.99, epochs=400, seed=42):
    actual_exp_name = f"{exp_name}_seed{seed}"
    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: {actual_exp_name}")
    print(f"Dataset: {dataset.upper()} | Target Sparsity: {sparsity} | Epochs: {epochs} | Seed: {seed}")
    
    # 準備合併後的環境變數
    run_env = os.environ.copy()
    run_env.update(env_vars)
    run_env["TARGET_DATASET"] = dataset
    run_env["TARGET_SPARSITY"] = str(sparsity)
    run_env["NUM_EPOCHS"] = str(epochs)
    run_env["RUN_SEED"] = str(seed)
    
    # 印出要被覆蓋設定的追蹤參數
    override_params = {**env_vars, "TARGET_DATASET": dataset, "TARGET_SPARSITY": sparsity, "NUM_EPOCHS": epochs, "RUN_SEED": seed}
    print(f"System Overrides: {override_params}")
    print(f"{'='*50}\n")
    
    # 將 log 存入專屬資料夾 (鎖定在 main/ 之下)
    ablation_log_dir = os.path.join(MAIN_DIR, "ablation_logs")
    os.makedirs(ablation_log_dir, exist_ok=True)
    log_file = os.path.join(ablation_log_dir, f"{actual_exp_name}.log")
    
    cmd = ["python", "-u", script]  # 動態指定執行的 Python 腳本
    
    try:
        with open(log_file, "w") as f:
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            
            while process.poll() is None:
                time.sleep(1)
                
            if process.returncode == 0:
                print(f"✅ Experiment '{actual_exp_name}' completed! Log saved to: {log_file}")
                
                # --- Linear Evaluation Step ---
                runs_dir = os.path.join(MAIN_DIR, "runs")
                runs = [os.path.join(runs_dir, d) for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))]
                if runs:
                    # 改用修改時間 (mtime) 排序，確保抓到的是「剛剛訓練完」的那個資料夾
                    runs.sort(key=os.path.getmtime)
                    latest_run = runs[-1]
                    # 按照使用者要求：不使用 best.pt (避免 Data Leakage)，統一使用最後一版 last.pt
                    encoder_path = os.path.join(latest_run, "last.pt")
                    
                    if not os.path.exists(encoder_path):
                        print(f"⚠️ Warning: {encoder_path} not found, skipping evaluation.")
                    else:
                        run_env["ENCODER_PATH"] = encoder_path
                        
                        # 統一使用新的標準評估腳本 (包含 CenterCrop, k=200 KNN, BatchNorm, L2 Norm)
                        eval_script = "evaluate_model.py" 
                        run_env["METHOD"] = "hebbian" if script == "Hebbian.py" else "rigl"
                        
                        print(f"🚀 Running Standardized Evaluation: {eval_script} on {encoder_path}")
                        eval_cmd = ["python", "-u", eval_script]
                        
                        with open(log_file, "a") as f_eval:
                            f_eval.write(f"\n\n{'='*50}\n--- Starting Standardized Evaluation (V8 Upgrade) ---\n{'='*50}\n")
                            subprocess.run(eval_cmd, env=run_env, stdout=f_eval, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
                            f_eval.write(f"\n\n{'='*50}\n--- End Evaluation ---\n{'='*50}\n")
                
            else:
                print(f"❌ Experiment '{actual_exp_name}' failed with return code {process.returncode}. Check log: {log_file}")
                
    except KeyboardInterrupt:
        print("\n⚠️ Experiment interrupted by user.")
        process.terminate()
        process.wait()


if __name__ == "__main__":
    run_mode = "all"
    if len(sys.argv) > 1:
        run_mode = sys.argv[1].lower()
    
    valid_modes = ["all", "hebbian", "rigl"]
    if run_mode not in valid_modes:
        print(f"Error: Invalid mode '{run_mode}'. Available modes are {valid_modes}")
        sys.exit(1)
        
    print("\n" + "*"*60)
    print(f"🌟 Welcome to Hebbian V10 Automated Defense Runner")
    print(f"🌟 Run Mode: {run_mode.upper()}")
    print("*"*60 + "\n")
    
    # 根據執行的腳本 (Hebbian.py 或 SimSiam.py) 過濾實驗
    def should_run(exp):
        if run_mode == "all": return True
        if run_mode == "hebbian" and exp.get("script", "Hebbian.py") == "Hebbian.py": return True
        if run_mode == "rigl" and exp.get("script", "Hebbian.py") == "SimSiam.py": return True
        return False
    
    # 1. 捍衛主戰場: 嚴格跑 3 Seeds
    CORE_DEFENSE_SEEDS = [42, 3407, 114514]
    for seed in CORE_DEFENSE_SEEDS:
        print(f"\n>>>>>> STARTING CORE DEFENSE SEED {seed} <<<<<<")
        for exp in GENERALIZATION:
            if should_run(exp):
                run_experiment(exp["name"], exp["env"], dataset=exp.get("dataset", "cifar100"), script=exp.get("script", "Hebbian.py"), sparsity=0.99, epochs=400, seed=seed)
            
    # 2. 曲線交叉點: 先跑 1 個 Seed 搶數據作圖 (未來如果有時間，改成 [42, 3407, 114514] 即可無縫接軌補完)
    CURVE_SEEDS = [42] 
    for seed in CURVE_SEEDS:
        print(f"\n>>>>>> STARTING SPARSITY CURVE CHECKPOINT SEED {seed} <<<<<<")
        for exp in SPARSITY_CURVE:
            if should_run(exp):
                run_experiment(exp["name"], exp["env"], dataset=exp.get("dataset", "cifar100"), script=exp.get("script", "Hebbian.py"), sparsity=exp.get("sparsity", 0.95), epochs=400, seed=seed)
        
    # # 3. 噪聲韌性挑戰: 只跑 1 個 Seed (42)
    # print(f"\n>>>>>> STARTING NOISE ROBUSTNESS CHALLENGE (SEED 42) <<<<<<")
    # for exp in NOISE_ROBUSTNESS:
    #     if should_run(exp):
    #         run_experiment(exp["name"], exp["env"], dataset="cifar100", script=exp.get("script", "Hebbian.py"), sparsity=0.99, epochs=400, seed=42)
            
    print(f"\n🎉 噪聲對抗實驗 '{run_mode.upper()}' 已全數完成！請檢查 ./ablation_logs/ 並開始撰寫論文！")
