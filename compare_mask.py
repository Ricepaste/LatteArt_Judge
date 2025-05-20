import torch
import os

baseline_masks_path = "baseline_rigl_initial_masks_0.9.pth"
my_masks_path = "my_rigl_initial_masks_0.9_generated.pth"

if not os.path.exists(baseline_masks_path):
    print(f"ERROR: Baseline masks file not found at {baseline_masks_path}")
    exit()
if not os.path.exists(my_masks_path):
    print(f"ERROR: My masks file not found at {my_masks_path}")
    exit()

print(f"Loading masks from {baseline_masks_path} and {my_masks_path}...")
baseline_masks = torch.load(baseline_masks_path)
my_masks = torch.load(my_masks_path)
print("Masks loaded.")

print("\n--- Comparing Masks ---")

# 獲取所有參數名稱的集合 (可能兩者管理的參數集合略有不同)
all_param_names = set(baseline_masks.keys()).union(set(my_masks.keys()))

num_diff_layers = 0
total_elements_compared = 0
total_diff_elements = 0

# 遍歷所有參數名稱進行比較
for name in sorted(list(all_param_names)):  # 按名稱排序，方便查看
    baseline_mask = baseline_masks.get(name)
    my_mask = my_masks.get(name)

    print(f"\nComparing layer: {name}")

    if baseline_mask is None and my_mask is None:
        print("  Both are None (dense or not managed). Identical.")
        continue

    if baseline_mask is None or my_mask is None:
        print(
            f"  Difference: One is None, the other is a tensor. Baseline None: {baseline_mask is None}, My Mask None: {my_mask is None}"
        )
        num_diff_layers += 1
        continue

    # 如果都是 tensor，進行詳細比較
    if baseline_mask.shape != my_mask.shape:
        print(
            f"  Difference: Shapes mismatch! Baseline: {baseline_mask.shape}, My Mask: {my_mask.shape}"
        )
        num_diff_layers += 1
        continue

    # 比較內容
    # 1. 稀疏度 (Non-zero counts)
    baseline_nonzero = torch.sum(baseline_mask).item()
    my_nonzero = torch.sum(my_mask).item()
    num_elements = baseline_mask.numel()
    total_elements_compared += num_elements

    baseline_sparsity = (
        1.0 - (baseline_nonzero / num_elements) if num_elements > 0 else 0.0
    )
    my_sparsity = 1.0 - (my_nonzero / num_elements) if num_elements > 0 else 0.0

    print(f"  Shape: {baseline_mask.shape}, Total elements: {num_elements}")
    print(f"  Baseline Nonzero: {baseline_nonzero} (Sparsity: {baseline_sparsity:.4f})")
    print(f"  My Mask Nonzero: {my_nonzero} (Sparsity: {my_sparsity:.4f})")

    if baseline_nonzero != my_nonzero:
        print(f"  Difference in nonzero count!")
        num_diff_layers += 1

    # 2. 具體的連接是否相同
    # 计算两个 mask 的差异： baseline 有但 my_mask 没有，或者 my_mask 有但 baseline 没有
    diff_tensor = baseline_mask ^ my_mask  # XOR operation (True where masks differ)
    num_diff_elements_in_layer = torch.sum(diff_tensor).item()
    total_diff_elements += num_diff_elements_in_layer

    if num_diff_elements_in_layer > 0:
        print(
            f"  Difference: Masks are not identical! Number of differing elements: {num_diff_elements_in_layer}/{num_elements} ({num_diff_elements_in_layer/num_elements:.2%})"
        )
        # 可以打印一些差异元素的比例，或者位置示例
        # print(f"  Difference ratio: {num_diff_elements_in_layer / num_elements:.4f}")
        if baseline_nonzero == my_nonzero:
            print("  Note: Nonzero counts match, but specific connections differ.")
        num_diff_layers += 1
    else:
        print("  Identical (element-wise).")

print("\n--- Comparison Summary ---")
print(f"Total layers compared: {len(all_param_names)}")
print(f"Number of layers with ANY difference: {num_diff_layers}")
print(
    f"Total elements compared across differing layers: {total_elements_compared}"
)  # 注意这里只累加了形状相同的层
if total_elements_compared > 0:
    print(
        f"Total differing elements: {total_diff_elements} ({total_diff_elements/total_elements_compared:.2%})"
    )
else:
    print("No comparable elements found.")
