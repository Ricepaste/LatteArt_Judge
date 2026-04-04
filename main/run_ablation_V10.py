import os
import subprocess
import time
import glob
import sys

# 取得 main/ 目錄的絕對路徑，確保所有的 relative paths 都不受執行位置的影響
MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Ablation Tests Configuration ---
# 1表示啟用, 0表示關閉
# 第一組: Ablation experiments at Sparsity 0.99
ABLATIONS = [
    {"name": "BASE_V4_OnlyAntiHebb_Entropy", "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "0", "ABLATION_ENTROPY": "1"}},
    {"name": "V8_OnlyVariance_Entropy", "env": {"ABLATION_ANTI_HEBB": "0", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}},
    {"name": "V8_Full", "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}}
]

# 第二組: Dataset Generalization Tests at Sparsity 0.99
GENERALIZATION = [
    {"name": "V8_Full_CIFAR100", "dataset": "cifar100", "script": "Hebbian.py", "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}},
    {"name": "RigL_Baseline_CIFAR100", "dataset": "cifar100", "script": "SimSiam.py", "env": {}},
    {"name": "RigL_Baseline_CIFAR10", "dataset": "cifar10", "script": "SimSiam.py", "env": {}}
]

# 第三組: Sparsity Degradation Curve (Performance vs Sparsity)
SPARSITY_CURVE = []
for s in [0.8, 0.9, 0.95, 0.99]:
    SPARSITY_CURVE.append({"name": f"Curve_V8_Full_{s}", "dataset": "cifar10", "script": "Hebbian.py", "sparsity": s, "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}})
    SPARSITY_CURVE.append({"name": f"Curve_RigL_{s}", "dataset": "cifar10", "script": "SimSiam.py", "sparsity": s, "env": {}})


def run_experiment(exp_name, env_vars, dataset="cifar10", script="Hebbian.py", sparsity=0.99, epochs=400):
    print(f"\n{'='*50}")
    print(f"🚀 Starting Experiment: {exp_name}")
    print(f"Dataset: {dataset.upper()} | Target Sparsity: {sparsity} | Epochs: {epochs}")
    
    # 準備合併後的環境變數
    run_env = os.environ.copy()
    run_env.update(env_vars)
    run_env["TARGET_DATASET"] = dataset
    run_env["TARGET_SPARSITY"] = str(sparsity)
    run_env["NUM_EPOCHS"] = str(epochs)
    
    # 確保 PYTHONPATH 包含 main/ 目錄，避免找不到 src匯入
    run_env["PYTHONPATH"] = MAIN_DIR + ":" + run_env.get("PYTHONPATH", "")
    
    # 印出要被覆蓋設定的追蹤參數
    override_params = {**env_vars, "TARGET_DATASET": dataset, "TARGET_SPARSITY": sparsity, "NUM_EPOCHS": epochs}
    print(f"System Overrides: {override_params}")
    print(f"{'='*50}\n")
    
    # 將 log 存入專屬資料夾 (鎖定在 main/ 之下)
    ablation_log_dir = os.path.join(MAIN_DIR, "ablation_logs")
    os.makedirs(ablation_log_dir, exist_ok=True)
    log_file = os.path.join(ablation_log_dir, f"{exp_name}.log")
    
    cmd = ["python", "-u", script]  # 動態指定執行的 Python 腳本
    
    
    try:
        with open(log_file, "w") as f:
            # 加入 cwd=MAIN_DIR 確保程式從 main/ 目錄執行
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            
            # 使用一個迴圈可以讓我們隨時按 Ctrl+C 中斷
            while process.poll() is None:
                time.sleep(1)
                
            if process.returncode == 0:
                print(f"✅ Experiment '{exp_name}' completed! Log saved to: {log_file}")
                
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
                print(f"❌ Experiment '{exp_name}' failed with return code {process.returncode}. Check log: {log_file}")
                
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
    
    # 1. Ablation Tests (99% Sparsity on CIFAR10)
    for exp in ABLATIONS:
        if should_run(exp):
            run_experiment(exp["name"], exp["env"], dataset="cifar10", script=exp.get("script", "Hebbian.py"), sparsity=0.99, epochs=400)
        
    # 2. Generalization Tests & Baselines (99% Sparsity)
    for exp in GENERALIZATION:
        if should_run(exp):
            run_experiment(exp["name"], exp["env"], dataset=exp.get("dataset", "cifar100"), script=exp.get("script", "Hebbian.py"), sparsity=0.99, epochs=400)
        
    # 3. Sparsity Degradation Curve Tests
    for exp in SPARSITY_CURVE:
        if should_run(exp):
            run_experiment(exp["name"], exp["env"], dataset=exp.get("dataset", "cifar10"), script=exp.get("script", "Hebbian.py"), sparsity=exp.get("sparsity", 0.99), epochs=400)
        
    print(f"\n🎉 All tests for mode '{run_mode.upper()}' finished! Please check ./ablation_logs/ for results.")
