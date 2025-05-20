import torch
import torchvision.models as models  # 或者你 SimSiam 模型依賴的其他庫
from rigl_torch.RigL import RigLScheduler  # 關鍵：導入原始的 RigLScheduler
import os

# --- 你需要根據你的情況修改下面的模型初始化部分 ---
# 目標是正確地創建你的 SimSiam 模型實例
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
simsiam_model = None
try:
    # 假設你的 SimSiam 模型定義在 src.module.SimSiam_Module 中，類名是 SimSiam
    # 並且它需要一個預訓練的 encoder base，比如 shufflenet_v2_x0_5
    from main.src.module.SimSiam_Module import (
        SimSiam,
    )  # <--- 修改這裡，如果你的路徑或類名不同

    pretrained_encoder_base = models.shufflenet_v2_x0_5(
        weights=None
    )  # <--- 修改這裡，如果你的 encoder base 不同
    simsiam_model = SimSiam(pretrained_encoder_base).to(device)
    print("SimSiam model created successfully for mask generation.")
except ImportError:
    print(
        "<<<<< 請修改這裡：無法導入你的 SimSiam 模型。請檢查上面的 from ... import ... 路徑。 >>>>>"
    )
    exit()
except Exception as e:
    print(f"<<<<< 請修改這裡：創建 SimSiam 模型時出錯：{e} >>>>>")
    exit()
# --- 模型初始化部分結束 ---

if simsiam_model:
    # 創建一個虛擬的優化器 (原始 RigL 初始化需要)
    optimizer = torch.optim.SGD(simsiam_model.parameters(), lr=0.01)
    print("Dummy optimizer created.")

    try:
        print(
            "Initializing ORIGINAL RigLScheduler with dense_allocation=0.1 (90% sparsity)..."
        )
        # 實例化原始的 RigLScheduler
        original_rigl_scheduler = RigLScheduler(
            model=simsiam_model,
            optimizer=optimizer,
            dense_allocation=0.1,  # 目標：90% 稀疏度
            T_end=1000,  # 這些參數對初始遮罩影響不大，設典型值即可
            delta=100,
            alpha=0.3,
            sparsity_distribution="uniform",  # 確保和你的實驗設定一致
            ignore_linear_layers=False,  # 確保和你的實驗設定一致
            # 注意：原始 RigL 的 get_W 通常只選取卷積和線性層的 weight
        )
        print("Original RigLScheduler initialized.")
        # print(str(original_rigl_scheduler)) # 可以取消註解這行，看看原始 RigL 的初始狀態

        # --- 從原始 RigL 中提取遮罩並保存為 name -> mask_tensor 的字典 ---
        masks_to_save = {}
        # 原始 RigL 的 self.W 是一個列表，存的是它管理的參數張量的引用
        # self.backward_masks 也是一個列表，與 self.W 對應

        # 我們需要將 self.W 中的張量映射回它們在模型中的名字
        # 這一步比較tricky，因為原始 RigL 是 index-based
        # 我們遍歷模型中所有帶名字的參數，然後看它是不是在原始 RigL 的 self.W 列表裡

        # 獲取原始 RigL 內部管理的參數列表 self.W (這是張量列表)
        # 和對應的遮罩列表 self.backward_masks
        original_W_list = original_rigl_scheduler.W
        original_masks_list = original_rigl_scheduler.backward_masks

        # 遍歷模型中所有帶名字的參數
        param_map_count = 0
        for param_idx_in_W, W_tensor_from_rigl in enumerate(original_W_list):
            found_name = None
            for name, model_param_tensor in simsiam_model.named_parameters():
                if model_param_tensor is W_tensor_from_rigl:  # 比較 Python 對象 ID
                    found_name = name
                    break

            if found_name:
                mask_tensor = original_masks_list[param_idx_in_W]
                if mask_tensor is not None:
                    masks_to_save[found_name] = mask_tensor.cpu()  # 保存到 CPU
                    # print(f"  Mapping mask for original RigL W index {param_idx_in_W} to name: {found_name}")
                    param_map_count += 1
                # else:
                # print(f"  Original RigL W index {param_idx_in_W} (name: {found_name}) is dense, no mask to save.")
            # else:
            # print(f"  Warning: Could not find name for original RigL W index {param_idx_in_W}")

        print(
            f"Successfully mapped and collected {param_map_count} masks from original RigL."
        )

        if masks_to_save:
            save_path = "baseline_rigl_initial_masks_0.9.pth"
            torch.save(masks_to_save, save_path)
            print(
                f"Initial masks from Baseline RigL (dense_alloc=0.1) saved to: {os.path.abspath(save_path)}"
            )
            print(f"Number of named masks saved: {len(masks_to_save)}")
            # 打印一下保存了哪些參數的遮罩，以及它們的稀疏度，方便檢查
            # print("Details of saved masks:")
            # for name, mask in masks_to_save.items():
            #     sparsity = 1.0 - mask.float().mean().item()
            #     print(f"  - {name}: Sparsity={sparsity:.4f}, Shape={mask.shape}")
        else:
            print(
                "ERROR: No masks were extracted and saved. Check the mapping logic or if original RigL produced any sparse layers."
            )

    except Exception as e:
        print(f"ERROR during original RigL mask generation: {e}")
        import traceback

        traceback.print_exc()

# --- 新增：計算並打印 Baseline RigL 的實際總體稀疏度 ---
print("\n--- Analyzing Actual Sparsity of Baseline RigL Initial Masks ---")
total_elements_baseline = 0
total_nonzero_baseline = 0

# 原始 RigL 的 self.W 存的是它管理的參數張量
# self.backward_masks 是與 self.W 對應的遮罩列表
for i, W_tensor_from_rigl in enumerate(original_rigl_scheduler.W):
    mask_tensor = original_rigl_scheduler.backward_masks[i]
    num_elements_layer = W_tensor_from_rigl.numel()  # 張量的總元素數
    total_elements_baseline += num_elements_layer

    if mask_tensor is not None:
        # 如果有遮罩，計算非零元素的數量
        num_nonzero_layer = torch.sum(mask_tensor).item()
        total_nonzero_baseline += num_nonzero_layer
        # print(f"  Layer index {i}: Nonzero={num_nonzero_layer}/{num_elements_layer}")
    else:
        # 如果沒有遮罩 (None)，表示是密集層，所有元素都算非零
        total_nonzero_baseline += num_elements_layer
        # print(f"  Layer index {i}: Dense, Nonzero={num_elements_layer}/{num_elements_layer}")

if total_elements_baseline > 0:
    actual_overall_sparsity_baseline = 1.0 - (
        total_nonzero_baseline / total_elements_baseline
    )
    print(
        f"Baseline RigL (Target Dense Allocation {original_rigl_scheduler.dense_allocation:.2f}) Actual Overall Sparsity: {actual_overall_sparsity_baseline:.4f}"
    )
else:
    print("No elements found in Baseline RigL's W list for sparsity calculation.")

print("--------------------------------------------------------------------\n")
