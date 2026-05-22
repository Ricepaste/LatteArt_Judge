# main/run_robustness_label_noise.py
import os
import argparse
import subprocess
import time
import pandas as pd
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

def run_evaluation(model_info, target_ds, noise_rate, fraction, epochs):
    print(f"\n>>>> [RUNNING] {model_info['name']} ({model_info['method']}) on {target_ds.upper()}")
    print(f"     [CONFIG] Label Noise Rate: {noise_rate*100:.1f}%, Data Fraction: {fraction*100:.1f}%, Epochs: {epochs}")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = target_ds
    run_env["NUM_EPOCHS"] = str(epochs)
    run_env["EVAL_FRACTION"] = str(fraction)
    run_env["LABEL_NOISE_RATE"] = str(noise_rate)
    run_env["SUMMARY_FILE_OVERRIDE"] = os.path.join(MAIN_DIR, "robustness_summary_master.csv")
    
    script = "evaluate_model.py"
    cmd = ["python", "-u", script]
    
    # Create timestamp log folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(MAIN_DIR, "robustness_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    log_filename = f"{model_info['method']}_{target_ds}_noise{int(noise_rate*100)}.log"
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
    parser = argparse.ArgumentParser(description="Automate Label Noise Robustness Test on Downstream Datasets")
    parser.add_argument("--hebbian_path", type=str, default="main/runs/Hebbian_SSL_20260410-190945/last.pt",
                        help="Path to Hebbian (Ours) pre-trained checkpoint (last.pt)")
    parser.add_argument("--rigl_path", type=str, default="main/runs/shuffleNet_v05_SimSiam__4/last.pt",
                        help="Path to RigL pre-trained checkpoint (last.pt)")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="Flip rate of training labels (default: 0.1)")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "svhn", "eurosat"], 
                        help="Datasets to evaluate (choices: cifar100, cifar10, svhn, eurosat, dtd, pcam)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of linear probing epochs (default: 50)")
    parser.add_argument("--eval_fraction", type=float, default=1.0, help="Few-shot label fraction (default: 1.0)")
    parser.add_argument("--clear_history", action="store_true", help="Clear past robustness summary file before run")
    
    args = parser.parse_args()
    
    # Path resolution (handle relative/absolute paths)
    hebbian_path = args.hebbian_path
    if not os.path.isabs(hebbian_path):
        hebbian_path = os.path.abspath(os.path.join(os.path.dirname(MAIN_DIR), hebbian_path))
        
    rigl_path = args.rigl_path
    if not os.path.isabs(rigl_path):
        rigl_path = os.path.abspath(os.path.join(os.path.dirname(MAIN_DIR), rigl_path))
        
    MODELS_TO_TEST = []
    if os.path.exists(hebbian_path):
        MODELS_TO_TEST.append({
            "name": "Ours (Hebbian)",
            "method": "hebbian",
            "encoder_path": hebbian_path
        })
    else:
        print(f"⚠️ Warning: Hebbian path not found at {hebbian_path}")
        
    if os.path.exists(rigl_path):
        MODELS_TO_TEST.append({
            "name": "RigL Baseline",
            "method": "rigl",
            "encoder_path": rigl_path
        })
    else:
        print(f"⚠️ Warning: RigL path not found at {rigl_path}")
        
    if not MODELS_TO_TEST:
        print("❌ Error: No valid pre-trained model checkpoints found. Please specify --hebbian_path or --rigl_path.")
        exit(1)
        
    master_csv = os.path.join(MAIN_DIR, "robustness_summary_master.csv")
    if args.clear_history and os.path.exists(master_csv):
        print(f"🧹 Clearing old robustness master CSV: {master_csv}")
        os.remove(master_csv)
        
    print("\n" + "="*70)
    print("🚀 AUTOMATED ROBUSTNESS EVALUATION SUITE STARTING (LABEL NOISE)")
    print(f"Target Noise Rate: {args.noise_rate * 100:.1f}%")
    print("="*70)
    
    start_time = time.time()
    
    for model in MODELS_TO_TEST:
        for ds in args.datasets:
            run_evaluation(model, ds.lower(), args.noise_rate, args.eval_fraction, args.epochs)
                
    duration = (time.time() - start_time) / 60
    print(f"\n🎉 All tests finished in {duration:.1f} minutes.")
    
    # Display comparison table
    if os.path.exists(master_csv):
        print("\n" + "📊 MASTER COMPARISON TABLE (LABEL NOISE) ".center(80, "="))
        df = pd.read_csv(master_csv)
        
        # Format noise rate and accuracy as percentages for clean printing
        df['Label_Noise_Rate'] = df['Label_Noise_Rate'].apply(lambda x: f"{x*100:.1f}%")
        df['KNN_Acc'] = df['KNN_Acc'].apply(lambda x: f"{x*100:.2f}%")
        df['Linear_Acc'] = df['Linear_Acc'].apply(lambda x: f"{x*100:.2f}%")
        
        print(df.tail(len(MODELS_TO_TEST) * len(args.datasets)).to_string(index=False))
        print("="*80)
        print(f"Full robustness data saved in: {master_csv}\n")
    else:
        print("\n❌ Error: No results collected in robustness_summary_master.csv")
