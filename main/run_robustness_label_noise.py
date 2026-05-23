# main/run_robustness_label_noise.py
import os
import argparse
import subprocess
import time
import pandas as pd
from datetime import datetime

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))

def run_evaluation(model_info, target_ds, noise_rate, fraction, epochs, shuffle_mask=False, random_prune_dense=False, sparsity=0.99):
    model_name = model_info['name']
    if shuffle_mask:
        model_name = f"{model_name} (Shuffled Mask)"
    elif random_prune_dense:
        model_name = f"{model_name} (Random Pruned)"
        
    print(f"\n>>>> [RUNNING] {model_name} ({model_info['method']}) on {target_ds.upper()}")
    print(f"     [CONFIG] Label Noise Rate: {noise_rate*100:.1f}%, Target Sparsity: {sparsity*100:.1f}%, Data Fraction: {fraction*100:.1f}%, Epochs: {epochs}")
    
    run_env = os.environ.copy()
    run_env["ENCODER_PATH"] = model_info["encoder_path"]
    run_env["METHOD"] = model_info["method"]
    run_env["TARGET_DATASET"] = target_ds
    run_env["NUM_EPOCHS"] = str(epochs)
    run_env["EVAL_FRACTION"] = str(fraction)
    run_env["LABEL_NOISE_RATE"] = str(noise_rate)
    run_env["TARGET_SPARSITY"] = str(sparsity)
    run_env["SHUFFLE_MASK"] = "True" if shuffle_mask else "False"
    run_env["RANDOM_PRUNE_DENSE"] = "True" if random_prune_dense else "False"
    run_env["SUMMARY_FILE_OVERRIDE"] = os.path.join(MAIN_DIR, "robustness_summary_master.csv")
    
    script = "evaluate_model.py"
    cmd = ["python", "-u", script]
    
    # Create timestamp log folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = os.path.join(MAIN_DIR, "robustness_logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    
    suffix = ""
    if shuffle_mask:
        suffix = "_shuffled"
    elif random_prune_dense:
        suffix = "_random_pruned"
        
    log_filename = f"{model_info['method']}{suffix}_{target_ds}_noise{int(noise_rate*100)}.log"
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
    parser.add_argument("--dense_path", type=str, default="", help="Path to Dense pre-trained checkpoint (last.pt)")
    parser.add_argument("--random_path", type=str, default="", help="Path to Random pre-trained checkpoint (last.pt)")
    parser.add_argument("--shuffle_mask", action="store_true", help="Evaluate Hebbian/Ours with shuffled mask as baseline")
    parser.add_argument("--random_prune_dense", action="store_true", help="Evaluate Dense/RigL model with random pruning mask as baseline")
    parser.add_argument("--sparsity", type=float, default=0.99, help="Target sparsity for pruning (default: 0.99)")
    parser.add_argument("--noise_rate", type=float, default=0.1, help="Flip rate of training labels (default: 0.1)")
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "svhn", "eurosat"], 
                        help="Datasets to evaluate (choices: cifar100, cifar10, svhn, eurosat, dtd, pcam)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of linear probing epochs (default: 50)")
    parser.add_argument("--eval_fraction", type=float, default=1.0, help="Few-shot label fraction (default: 1.0)")
    parser.add_argument("--clear_history", action="store_true", help="Clear past robustness summary file before run")
    
    args = parser.parse_args()
    
    # Path resolution (handle relative/absolute paths)
    hebbian_path = args.hebbian_path
    if hebbian_path and hebbian_path.lower() not in ["imagenet", "official"] and not os.path.isabs(hebbian_path):
        hebbian_path = os.path.abspath(os.path.join(os.path.dirname(MAIN_DIR), hebbian_path))
        
    rigl_path = args.rigl_path
    if rigl_path and rigl_path.lower() not in ["imagenet", "official"] and not os.path.isabs(rigl_path):
        rigl_path = os.path.abspath(os.path.join(os.path.dirname(MAIN_DIR), rigl_path))

    dense_path = args.dense_path
    if dense_path and dense_path.lower() not in ["imagenet", "official"] and not os.path.isabs(dense_path):
        dense_path = os.path.abspath(os.path.join(os.path.dirname(MAIN_DIR), dense_path))

    random_path = args.random_path
    if random_path and random_path.lower() not in ["imagenet", "official"] and not os.path.isabs(random_path):
        random_path = os.path.abspath(os.path.join(os.path.dirname(MAIN_DIR), random_path))
        
    MODELS_TO_TEST = []
    if hebbian_path and (hebbian_path.lower() in ["imagenet", "official"] or os.path.exists(hebbian_path)):
        name_label = "Ours (Hebbian)" if hebbian_path.lower() not in ["imagenet", "official"] else "ImageNet Pretrained"
        MODELS_TO_TEST.append({
            "name": name_label,
            "method": "hebbian",
            "encoder_path": hebbian_path
        })
    else:
        if hebbian_path:
            print(f"⚠️ Warning: Hebbian path not found at {hebbian_path}")
        
    if rigl_path and (rigl_path.lower() in ["imagenet", "official"] or os.path.exists(rigl_path)):
        name_label = "RigL Baseline" if rigl_path.lower() not in ["imagenet", "official"] else "ImageNet Pretrained"
        MODELS_TO_TEST.append({
            "name": name_label,
            "method": "rigl",
            "encoder_path": rigl_path
        })
    else:
        if rigl_path:
            print(f"⚠️ Warning: RigL path not found at {rigl_path}")

    if dense_path and (dense_path.lower() in ["imagenet", "official"] or os.path.exists(dense_path)):
        name_label = "Fully Dense" if dense_path.lower() not in ["imagenet", "official"] else "ImageNet Pretrained"
        MODELS_TO_TEST.append({
            "name": name_label,
            "method": "rigl",
            "encoder_path": dense_path
        })
    else:
        if dense_path:
            print(f"⚠️ Warning: Dense path not found at {dense_path}")

    if random_path and (random_path.lower() in ["imagenet", "official"] or os.path.exists(random_path)):
        name_label = "Random Pruning" if random_path.lower() not in ["imagenet", "official"] else "ImageNet Pretrained"
        MODELS_TO_TEST.append({
            "name": name_label,
            "method": "random",
            "encoder_path": random_path
        })
    else:
        if random_path:
            print(f"⚠️ Warning: Random path not found at {random_path}")
        
    if not MODELS_TO_TEST:
        print("❌ Error: No valid pre-trained model checkpoints found. Please specify paths.")
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
    total_runs = 0
    
    for model in MODELS_TO_TEST:
        for ds in args.datasets:
            # 1. 執行標準模型評估
            run_evaluation(model, ds.lower(), args.noise_rate, args.eval_fraction, args.epochs, sparsity=args.sparsity)
            total_runs += 1
            
            # 2. 若為 Hebbian (Ours) 且啟用 shuffle_mask，執行打亂遮罩評估
            if model["method"] == "hebbian" and args.shuffle_mask:
                run_evaluation(model, ds.lower(), args.noise_rate, args.eval_fraction, args.epochs, shuffle_mask=True, sparsity=args.sparsity)
                total_runs += 1
                
            # 3. 若為 RigL/Dense 且啟用 random_prune_dense，執行隨機剪枝評估
            if model["method"] == "rigl" and args.random_prune_dense:
                run_evaluation(model, ds.lower(), args.noise_rate, args.eval_fraction, args.epochs, random_prune_dense=True, sparsity=args.sparsity)
                total_runs += 1
                
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
        
        print(df.tail(total_runs).to_string(index=False))
        print("="*80)
        print(f"Full robustness data saved in: {master_csv}\n")
    else:
        print("\n❌ Error: No results collected in robustness_summary_master.csv")
