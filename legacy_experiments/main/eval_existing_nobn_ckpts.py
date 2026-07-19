import os
import subprocess

MAIN_DIR = os.path.dirname(os.path.abspath(__file__))
ours_ckpt = os.path.join(MAIN_DIR, "runs/NoBN_Backbone_Ours_Lateral_Inhibition/last.pt")
pos_ckpt = os.path.join(MAIN_DIR, "runs/NoBN_Backbone_Positive_Hebbian_Ablation/last.pt")

def run_eval(ckpt_path, name):
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at: {ckpt_path}")
        return "N/A", "N/A"
        
    print(f"\nEvaluating {name}...")
    env = os.environ.copy()
    env["ENCODER_PATH"] = ckpt_path
    env["METHOD"] = "hebbian"
    env["TARGET_DATASET"] = "cifar100"
    env["NUM_EPOCHS"] = "50"  # We run standard 50 epochs for stable probing
    env["TARGET_SPARSITY"] = "0.99"
    env["BACKBONE_NO_BN"] = "True"  # Crucial! Prevents re-creating BN layers during evaluation
    env["DISABLE_CLASSIFIER_BN"] = "False"  # Re-enable classifier BN
    
    process = subprocess.Popen(
        ["python", "-u", "evaluate_model.py"],
        env=env,
        cwd=MAIN_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    stdout, _ = process.communicate()
    
    linear_acc = "N/A"
    knn_acc = "N/A"
    for line in stdout.split("\n"):
        if "Final Linear Probing Accuracy:" in line:
            linear_acc = line.split(":")[-1].strip()
        elif "KNN Protocol Accuracy" in line:
            knn_acc = line.split(":")[-1].strip()
            
    print(f"  > KNN Accuracy: {knn_acc}")
    print(f"  > Linear Probing Accuracy: {linear_acc}")
    return knn_acc, linear_acc

if __name__ == "__main__":
    print("="*80)
    print("🔬 EVALUATING EXISTING NO-BN BACKBONE CHECKPOINTS (WITH CLASSIFIER BN)")
    print("="*80)
    
    run_eval(ours_ckpt, "Ours (Lateral Inhibition)")
    run_eval(pos_ckpt, "Positive Hebbian Ablation")
    print("\n" + "="*80 + "\n")
