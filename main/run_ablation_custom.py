# main/run_ablation_custom.py
import os
import subprocess
import time
import glob
from datetime import datetime

def find_latest_hebbian_run(runs_dir):
    folders = glob.glob(os.path.join(runs_dir, "Hebbian_SSL_*"))
    if not folders:
        return None
    folders.sort(key=os.path.getmtime)
    return folders[-1]

def main():
    repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    main_dir = os.path.join(repo_dir, "main")
    runs_dir = os.path.join(main_dir, "runs")
    
    # 確保輸出目錄存在
    os.makedirs(runs_dir, exist_ok=True)
    summary_file = os.path.join(runs_dir, "custom_ablation_summary.csv")
    
    # 清理舊的摘要檔
    if os.path.exists(summary_file):
        os.remove(summary_file)

    # 定義消融實驗配置 (對比各組件的貢獻)
    ABLATION_CONFIGS = [
        # {
        #     "name": "w_o_AntiHebbian",
        #     "env": {"ABLATION_ANTI_HEBB": "0", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}
        # },
        # {
        #     "name": "w_o_Entropy",
        #     "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "0"}
        # },
        # {
        #     "name": "w_o_Variance",
        #     "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "0", "ABLATION_ENTROPY": "1"}
        # }
        {
            "name": "all",
            "env": {"ABLATION_ANTI_HEBB": "1", "ABLATION_VARIANCE": "1", "ABLATION_ENTROPY": "1"}
        }
    ]

    print("\n" + "=" * 60)
    print("🧪 CUSTOM ABLATION STUDY RUNNER STARTING")
    print("   - Pre-training: 50 Epochs")
    print("   - Linear Probing: 30 Epochs")
    print("   - Dataset: CIFAR-100")
    print("   - Sparsity: 99%")
    print("=" * 60 + "\n")

    for idx, config in enumerate(ABLATION_CONFIGS):
        print(f"\n[#{idx+1}/{len(ABLATION_CONFIGS)}] Running Configuration: {config['name']}")
        print(f"    Settings: {config['env']}")
        
        # ----------------- 1. 預訓練階段 (50 Epochs) -----------------
        print("    >> Starting 50 Epochs Pre-training...")
        train_env = os.environ.copy()
        train_env.update(config["env"])
        train_env["TARGET_DATASET"] = "cifar100"
        train_env["NUM_EPOCHS"] = "50"
        train_env["TARGET_SPARSITY"] = "0.99"
        train_env["RUN_SEED"] = "42"
        
        log_dir = os.path.join(main_dir, "ablation_logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"custom_ablation_{config['name']}.log")
        
        train_cmd = ["python", "-u", "Hebbian.py"]
        
        start_train = time.time()
        try:
            with open(log_file, "w") as f_log:
                f_log.write(f"=== pre-training log for {config['name']} ===\n")
                process = subprocess.Popen(train_cmd, env=train_env, stdout=f_log, stderr=subprocess.STDOUT, cwd=main_dir)
                process.wait()
            
            if process.returncode != 0:
                print(f"    ❌ Pre-training failed with code {process.returncode}. Check log: {log_file}")
                continue
                
            train_duration = (time.time() - start_train) / 60
            print(f"    ✅ Pre-training completed in {train_duration:.1f} mins.")
        except Exception as e:
            print(f"    ❌ Pre-training encountered error: {e}")
            continue

        # 偵測剛剛產生的權重檔
        latest_run = find_latest_hebbian_run(runs_dir)
        if not latest_run:
            print("    ❌ Failed to locate the saved weight folder.")
            continue
            
        encoder_path = os.path.join(latest_run, "last.pt")
        if not os.path.exists(encoder_path):
            print(f"    ❌ Checkpoint file not found: {encoder_path}")
            continue

        # ----------------- 2. 線性探測與 KNN 評估階段 (30 Epochs) -----------------
        print(f"    >> Starting 30 Epochs Evaluation on {encoder_path}...")
        eval_env = os.environ.copy()
        eval_env["ENCODER_PATH"] = encoder_path
        eval_env["METHOD"] = "hebbian"
        eval_env["TARGET_DATASET"] = "cifar100"
        eval_env["NUM_EPOCHS"] = "30"
        eval_env["TARGET_SPARSITY"] = "0.99"
        eval_env["SUMMARY_FILE_OVERRIDE"] = summary_file
        
        eval_cmd = ["python", "-u", "evaluate_model.py"]
        
        try:
            with open(log_file, "a") as f_log:
                f_log.write(f"\n\n=== evaluation log for {config['name']} ===\n")
                result = subprocess.run(eval_cmd, env=eval_env, stdout=f_log, stderr=subprocess.STDOUT, cwd=main_dir, check=True)
                
            # 解析剛寫入 CSV 的結果並印出
            if os.path.exists(summary_file):
                with open(summary_file, "r") as sf:
                    lines = sf.readlines()
                    if len(lines) > 1:
                        last_line = lines[-1].strip().split(",")
                        if len(last_line) >= 7:
                            knn_acc = float(last_line[5]) * 100
                            linear_acc = float(last_line[6]) * 100
                            print(f"    ✅ Finished | KNN Acc: {knn_acc:.2f}% | Linear Acc: {linear_acc:.2f}%")
        except Exception as e:
            print(f"    ❌ Evaluation failed: {e}")

    # 4. 印出最終總結
    if os.path.exists(summary_file):
        print("\n" + "=" * 60)
        print("📊 FINAL CUSTOM ABLATION STUDY RESULTS")
        print("=" * 60)
        with open(summary_file, "r") as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                # 替換模型資料夾名稱為易讀的名稱
                if idx > 0:
                    parts = line.strip().split(",")
                    if len(parts) >= 7:
                        # parts[0]: Method, parts[1]: ModelName (Hebbian_SSL_xxxxx)
                        # 我們將其對應到 ABLATION_CONFIGS 的名稱
                        config_name = ABLATION_CONFIGS[idx-1]["name"]
                        knn_pct = float(parts[5]) * 100
                        linear_pct = float(parts[6]) * 100
                        print(f"{config_name:<20} | KNN Acc: {knn_pct:.2f}% | Linear Acc: {linear_pct:.2f}%")
                else:
                    print(f"{'Configuration':<20} | KNN Accuracy  | Linear Probing Accuracy")
                    print("-" * 60)
        print("=" * 60)
        print(f"Summary saved in: {summary_file}")
    else:
        print("\n❌ Ablation study finished but no summary was collected.")

if __name__ == "__main__":
    main()
