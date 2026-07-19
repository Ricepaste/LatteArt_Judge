import os
import subprocess
import time
import pandas as pd
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_TO_TEST = [
    {
        "name": "Ours (Lateral Inhibition)",
        "method": "hebbian",
        "encoder_path": "/app/main/runs/Hebbian_SSL_20260410-190945/last.pt"
    },
    {
        "name": "Positive Hebbian Ablation",
        "method": "hebbian",
        "encoder_path": "/app/main/runs/Hebbian_SSL_20260612-231515/last.pt"
    }
]

TARGET_DATASETS = ["cifar100", "cifar10", "svhn", "stl10", "eurosat", "dtd", "pcam"]

def run_evaluation(model_info, target_ds):
    print(f"\n>>>> [RUNNING] {model_info['name']} on {target_ds.upper()} (No BN Mode)")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = target_ds
    run_env["NUM_EPOCHS"] = "50"
    run_env["TARGET_SPARSITY"] = "0.99"
    run_env["EVAL_FRACTION"] = "1.0"
    run_env["INPUT_NOISE_STD"] = "0.0"
    run_env["DISABLE_CLASSIFIER_BN"] = "True" # Force disabling of classifier's internal BN
    
    temp_summary_csv = os.path.join(MAIN_DIR, "temp_nobn_run.csv")
    run_env["SUMMARY_FILE_OVERRIDE"] = temp_summary_csv
    
    script = "evaluate_model.py"
    cmd = ["python", "-u", script]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(MAIN_DIR, "eval_logs_nobn", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{model_info['name'].replace(' ', '_')}_{target_ds}_nobn.log"
    log_path = os.path.join(log_dir, log_filename)
    
    try:
        with open(log_path, "w") as f:
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ Success. Log: {log_path}")
            if os.path.exists(temp_summary_csv):
                df = pd.read_csv(temp_summary_csv)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    return {
                        "Model": model_info["name"],
                        "Dataset": target_ds,
                        "Linear_Acc": f"{last_row['Linear_Acc']*100:.2f}%",
                        "raw_acc": last_row['Linear_Acc']
                    }
        else:
            print(f"❌ Failed. Check: {log_path}")
            
    except Exception as e:
        print(f"❌ Exception: {e}")
    return None

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 AUTOMATED ALL-DATASET EVALUATION (NO DOWNSTREAM BN MODE)")
    print("  (Evaluating representation generalizability without BatchNorm scaling)")
    print("="*80)
    
    temp_summary_csv = os.path.join(MAIN_DIR, "temp_nobn_run.csv")
    if os.path.exists(temp_summary_csv):
        os.remove(temp_summary_csv)
        
    start_time = time.time()
    results = []
    
    for ds in TARGET_DATASETS:
        for model in MODELS_TO_TEST:
            res = run_evaluation(model, ds)
            if res:
                results.append(res)
                
    duration = (time.time() - start_time) / 60
    print(f"\n🎉 All tests finished in {duration:.1f} minutes.")
    
    if results:
        # Build comparison table
        comp_data = []
        for ds in TARGET_DATASETS:
            ours_res = next((r for r in results if r["Model"] == "Ours (Lateral Inhibition)" and r["Dataset"] == ds), None)
            pos_res = next((r for r in results if r["Model"] == "Positive Hebbian Ablation" and r["Dataset"] == ds), None)
            
            ours_acc = ours_res["Linear_Acc"] if ours_res else "N/A"
            pos_acc = pos_res["Linear_Acc"] if pos_res else "N/A"
            
            gap = ""
            if ours_res and pos_res:
                gap = f"{(ours_res['raw_acc'] - pos_res['raw_acc'])*100:+.2f}%"
                
            comp_data.append({
                "Dataset": ds.upper(),
                "Ours Linear (No BN)": ours_acc,
                "PosHebb Linear (No BN)": pos_acc,
                "Gap (Ours - PosHebb)": gap
            })
            
        df_comp = pd.DataFrame(comp_data)
        print("\n" + "⚖️ SIDE-BY-SIDE NO-BN PERFORMANCE COMPARISON ".center(85, "="))
        print(df_comp.to_string(index=False))
        print("="*85 + "\n")
        
        # Clean up
        if os.path.exists(temp_summary_csv):
            os.remove(temp_summary_csv)
    else:
        print("❌ No results collected.")
