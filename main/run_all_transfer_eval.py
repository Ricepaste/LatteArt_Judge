import os
import subprocess
import time
import pandas as pd
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_TO_TEST = [
    {
        "name": "Positive Hebbian Ablation",
        "method": "hebbian",
        "encoder_path": "/app/main/runs/Hebbian_SSL_20260612-231515/last.pt"
    }
]

TARGET_DATASETS = ["cifar100", "cifar10", "svhn", "stl10", "eurosat", "dtd", "pcam"]

def run_evaluation(model_info, target_ds):
    print(f"\n>>>> [RUNNING] {model_info['name']} on {target_ds.upper()}")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = target_ds
    run_env["NUM_EPOCHS"] = "50"
    run_env["TARGET_SPARSITY"] = "0.99"
    run_env["EVAL_FRACTION"] = "1.0"
    run_env["INPUT_NOISE_STD"] = "0.0"
    
    # We override SUMMARY_FILE_OVERRIDE to a custom run CSV to avoid cluttering master CSV
    temp_summary_csv = os.path.join(MAIN_DIR, "temp_transfer_run.csv")
    run_env["SUMMARY_FILE_OVERRIDE"] = temp_summary_csv
    
    script = "evaluate_model.py"
    cmd = ["python", "-u", script]
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(MAIN_DIR, "eval_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{model_info['name'].replace(' ', '_')}_{target_ds}.log"
    log_path = os.path.join(log_dir, log_filename)
    
    try:
        with open(log_path, "w") as f:
            process = subprocess.Popen(cmd, env=run_env, stdout=f, stderr=subprocess.STDOUT, cwd=MAIN_DIR)
            process.wait()
            
        if process.returncode == 0:
            print(f"✅ Success. Log: {log_path}")
            # Read the last line appended to temp_summary_csv
            if os.path.exists(temp_summary_csv):
                df = pd.read_csv(temp_summary_csv)
                if len(df) > 0:
                    last_row = df.iloc[-1]
                    return {
                        "Model": model_info["name"],
                        "Dataset": target_ds,
                        "KNN_Acc": f"{last_row['KNN_Acc']*100:.2f}%",
                        "Linear_Acc": f"{last_row['Linear_Acc']*100:.2f}%"
                    }
        else:
            print(f"❌ Failed. Check: {log_path}")
            
    except Exception as e:
        print(f"❌ Exception running evaluation: {e}")
    return None

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 AUTOMATED ALL-DATASET EVALUATION SUITE STARTING")
    print("="*70)
    
    # Clean temp file if exists
    temp_summary_csv = os.path.join(MAIN_DIR, "temp_transfer_run.csv")
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
        print("\n" + "📊 POSITIVE HEBBIAN RUN SUMMARY ".center(75, "="))
        df_res = pd.DataFrame(results)
        print(df_res.to_string(index=False))
        print("="*75 + "\n")
        
        # Load master CSV to find Ours (Hebbian_SSL_20260410-190945)
        master_csv = os.path.join(MAIN_DIR, "transfer_summary_master.csv")
        if os.path.exists(master_csv):
            try:
                df_master = pd.read_csv(master_csv)
                # Filter for Ours (Data_Fraction = 1.0)
                df_ours = df_master[(df_master["Source_Model"] == "Hebbian_SSL_20260410-190945") & (df_master["Data_Fraction"] == 1.0)]
                
                # Build side-by-side comparison
                comparison = []
                for res in results:
                    dataset = res["Dataset"]
                    pos_knn = res["KNN_Acc"]
                    pos_linear = res["Linear_Acc"]
                    
                    # Find Ours matching row
                    ours_row = df_ours[df_ours["Target_Dataset"] == dataset]
                    if len(ours_row) > 0:
                        # Handle potential raw floats or already formatted strings
                        o_knn = ours_row.iloc[0]['KNN_Acc']
                        o_lin = ours_row.iloc[0]['Linear_Acc']
                        ours_knn = f"{o_knn*100:.2f}%" if isinstance(o_knn, (int, float)) else str(o_knn)
                        ours_linear = f"{o_lin*100:.2f}%" if isinstance(o_lin, (int, float)) else str(o_lin)
                    else:
                        ours_knn = "N/A"
                        ours_linear = "N/A"
                        
                    comparison.append({
                        "Dataset": dataset.upper(),
                        "Ours KNN": ours_knn,
                        "Ours Linear": ours_linear,
                        "PosHebb KNN": pos_knn,
                        "PosHebb Linear": pos_linear
                    })
                
                df_comp = pd.DataFrame(comparison)
                print("\n" + "⚖️ SIDE-BY-SIDE PERFORMANCE COMPARISON (Ours vs Positive Hebbian) ".center(85, "="))
                print(df_comp.to_string(index=False))
                print("="*85 + "\n")
            except Exception as e:
                print(f"⚠️ Could not build comparison table: {e}")
        
        # Also clean up the temp csv at the end
        if os.path.exists(temp_summary_csv):
            os.remove(temp_summary_csv)
    else:
        print("❌ No results collected.")
