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

# 💀 Priority 1: 捍衛主戰場 (CIFAR-100 @ 99%)
# 這是你論文的大絕招，必須跑滿 3 個 Seed 去堵住教授的嘴。
# -> [總共 2 個配置 x 3 Seeds = 6 個實驗] -> 分給兩台主機跑，約 6 天完成。
GENERALIZATION = [
    {"name": "V8_Full_CIFAR100_99", "dataset": "cifar100", "script": "Hebbian.py", "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}},
    {"name": "RigL_Baseline_CIFAR100_99", "dataset": "cifar100", "script": "SimSiam.py", "env": {}}
]

# ⚔️ Priority 2: 點出衰減交叉點 (CIFAR-100 @ 95%) 
# 放棄 0.8, 0.9。我們直接在 RigL 會開始崩潰的懸崖邊緣 (95%) 各打一個點！
# 單純為了畫出「黃金交叉圖」，這兩個點我們只跑 1 個 Seed (Seed 42) 就好。
# -> [總共 2 個實驗] -> 約 2 天完成。
SPARSITY_CURVE = []
for s in [0.8, 0.9, 0.95]:
    SPARSITY_CURVE.append({"name": f"Curve_V8_Full_{s}_C100", "dataset": "cifar100", "script": "Hebbian.py", "sparsity": s, "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}})
    SPARSITY_CURVE.append({"name": f"Curve_RigL_{s}_C100", "dataset": "cifar100", "script": "SimSiam.py", "sparsity": s, "env": {}})

# 🛡️ Priority 3: 舊資料防禦 (Ablation Tests)
# 表三跟表一的數據你已經有 2 次了！口試時直接用那兩次的數據取平均，不需要再拿寶貴的 GPU 去跑。
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
                runs = sorted(glob.glob(os.path.join(runs_dir, "*")))
                if runs:
                    latest_run = runs[-1]
                    # If best.pt exists, use it, else last.pt
                    encoder_path = os.path.join(latest_run, "best.pt")
                    if not os.path.exists(encoder_path):
                        encoder_path = os.path.join(latest_run, "last.pt")
                        
                    run_env["ENCODER_PATH"] = encoder_path
                    
                    eval_script = "Hebbian_linear_evaluation.py" if script == "Hebbian.py" else "SimSiam_linear_evaluation.py"
                    print(f"🚀 Running Linear Evaluation: {eval_script} on {encoder_path}")
                    eval_cmd = ["python", "-u", eval_script]
                    
                    with open(log_file, "a") as f_eval:
                        f_eval.write(f"\n\n{'='*50}\n--- Starting Linear Evaluation ---\n{'='*50}\n")
                        subprocess.run(eval_cmd, env=run_env, stdout=f_eval, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
                        f_eval.write(f"\n\n{'='*50}\n--- End Linear Evaluation ---\n{'='*50}\n")
                
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
        
    print(f"\n🎉 畢業生存任務 '{run_mode.upper()}' 已全數完成！請檢查 ./ablation_logs/ 並開始撰寫論文！")
