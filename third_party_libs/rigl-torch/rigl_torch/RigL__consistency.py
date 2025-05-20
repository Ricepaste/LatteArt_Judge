# RigL__consistency.py (Robustified Name-Based Version)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import warnings

# Assuming rigl_torch.util is in your python path and contains get_W
# from rigl_torch.util import get_W # get_W might not be needed if strictly name-based


class IndexMaskHook:  # Renaming to ParameterNameMaskHook for clarity might be good
    """Hook for RigL Scheduler. Stores parameter name."""

    def __init__(self, param_name, scheduler):
        self.param_name = param_name
        self.scheduler = scheduler
        # dense_grad is not used for grow scores in this consistency version,
        # as w.grad is used directly. Kept for potential debugging or other uses.
        self.dense_grad = None

    def __name__(self):
        return "ParameterNameMaskHook"  # Or IndexMaskHook if you prefer

    @torch.no_grad()
    def __call__(self, grad):
        # Access mask using parameter name from the scheduler's dictionary
        mask = self.scheduler.backward_masks.get(self.param_name)

        if mask is None:  # Layer is dense or not managed by RigL's sparsity
            return grad

        if mask.shape != grad.shape:
            warnings.warn(
                f"Mask shape {mask.shape} mismatch with grad shape {grad.shape} in Hook for '{self.param_name}'. Returning original grad.",
                RuntimeWarning,
            )
            return grad

        # Apply the mask to the gradient
        # This is the primary function of the hook in RigL: to zero out gradients for pruned weights
        masked_grad = grad * mask

        # --- Optional: Accumulate dense_grad (masked) if needed by some logic ---
        # Note: Original RigL used dense_grad (accumulated from unmasked input `grad` in hook) for grow scores.
        # This version uses w.grad directly in _rigl_step.
        # If check_if_backward_hook_should_accumulate_grad is true, this will run.
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                self.dense_grad = torch.zeros_like(masked_grad)  # Initialize with zeros

            if self.dense_grad.shape == masked_grad.shape:
                # grad_accumulation_n is forced to 1 in this scheduler's __init__
                self.dense_grad += masked_grad  # Effectively, dense_grad becomes the current masked_grad
            else:
                warnings.warn(
                    f"dense_grad shape mismatch in Hook for '{self.param_name}'. Resetting dense_grad.",
                    RuntimeWarning,
                )
                self.dense_grad = torch.zeros_like(masked_grad)
        else:
            self.dense_grad = None  # Reset if not accumulating

        return masked_grad


def _create_step_wrapper(scheduler, optimizer):
    """Wraps optimizer.step() to include RigL maintenance."""
    _unwrapped_step = optimizer.step

    def _wrapped_step(*args, **kwargs):
        # Call the original optimizer's step
        result = _unwrapped_step(*args, **kwargs)

        # Perform RigL maintenance operations AFTER the optimizer has updated weights
        scheduler.reset_momentum()  # Mask momentum buffer
        scheduler.apply_mask_to_weights()  # Ensure weights adhere to the mask
        # Original RigL also called apply_mask_to_gradients() inside _rigl_step.
        # We'll add it there for consistency.
        return result

    optimizer.step = _wrapped_step


class RigLScheduler:
    def __init__(
        self,
        model,
        optimizer,
        dense_allocation=0.1,  # Default to a sparse value for testing
        T_end=1000,  # Total steps for cosine annealing of alpha
        sparsity_distribution="uniform",  # 'uniform' keeps first layer dense
        ignore_linear_layers=False,  # Original RigL default was False for this param name. Let's be explicit.
        delta=100,  # Interval for RigL updates
        alpha=0.3,  # Fraction of connections to drop/grow
        consistency_lambda=0.0,  # Default to 0 (no consistency effect)
        static_topo=False,  # If true, no dynamic pruning/growing
        grad_accumulation_n=1,  # Forced to 1, as w.grad is used directly
        state_dict=None,  # For resuming from a saved state
    ):
        if not (0 < dense_allocation <= 1):  # dense_allocation is 1.0-sparsity
            raise ValueError(
                f"Dense allocation must be in (0, 1], got {dense_allocation}"
            )

        self.model = model
        self.optimizer = optimizer
        self.dense_allocation = dense_allocation
        self.static_topo = static_topo
        self.consistency_lambda = consistency_lambda
        self.grad_accumulation_n = (
            1  # Force to 1, as w.grad is used, not accumulated hook grad
        )
        self.sparsity_distribution = sparsity_distribution
        self.ignore_linear_layers = ignore_linear_layers

        # --- Name-based state dictionaries ---
        self.N = {}  # Number of elements for each managed parameter {name: count}
        self.S = (
            {}
        )  # Target sparsity level for each managed parameter {name: sparsity_float (0 to 1)}
        self.backward_masks = (
            {}
        )  # Current binary mask for each parameter {name: mask_tensor or None}
        self.is_linear = (
            {}
        )  # Flags if a parameter belongs to a Linear layer {name: bool}
        self._param_names = []  # Ordered list of managed parameter names

        self._init_sparsity_and_masks()  # Initialize S, N, backward_masks, etc.

        if state_dict is not None:
            self.load_state_dict(state_dict)
            # Ensure weights conform to loaded masks immediately
            self.apply_mask_to_weights()
        else:
            self.step = 0  # Current training step
            self.rigl_steps = 0  # Number of times RigL update has occurred
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end
            # Create initial random masks based on self.S
            self.random_sparsify_model()

        # --- 保存当前 RigL 生成的初始遮罩 ---
        generated_masks = {}
        for name, mask_tensor in self.backward_masks.items():
            if mask_tensor is not None:
                generated_masks[name] = mask_tensor.cpu()
        torch.save(generated_masks, "my_rigl_initial_masks_0.9_generated.pth")
        print(
            "My RigL generated initial masks saved to my_rigl_initial_masks_0.9_generated.pth"
        )

        print("\nDEBUG: Initial masks generated by RigL__consistency.py:")
        total_params_model = 0
        total_nonzero_in_masks = 0
        for name, param in self.model.named_parameters():
            if name in self.backward_masks:
                mask = self.backward_masks[name]
                s_target = self.S.get(name, 0.0)
                n_elements = self.N.get(name, 0)
                total_params_model += n_elements

                if mask is not None:
                    num_ones = torch.sum(mask).item()
                    total_nonzero_in_masks += num_ones
                    actual_sparsity = (
                        1.0 - (num_ones / n_elements) if n_elements > 0 else 0.0
                    )
                    print(
                        f"  Layer: {name}, Target Sparsity S: {s_target:.4f}, Actual Sparsity in Mask: {actual_sparsity:.4f}, Num_ones: {num_ones}/{n_elements}"
                    )
                    # 檢查權重是否真的被遮罩了
                    num_zeros_in_param = torch.sum(param.data == 0).item()
                    print(
                        f"    └─ Num zeros in param.data after initial sparsify: {num_zeros_in_param}"
                    )
                else:
                    total_nonzero_in_masks += n_elements  # Dense layer
                    print(
                        f"  Layer: {name}, Target Sparsity S: {s_target:.4f} (Dense Layer - Mask is None)"
                    )

        overall_sparsity_from_masks = (
            1.0 - (total_nonzero_in_masks / total_params_model)
            if total_params_model > 0
            else 0.0
        )
        print(
            f"Overall sparsity from generated masks: {overall_sparsity_from_masks:.4f} (Target dense_allocation implies sparsity of {1.0 - self.dense_allocation:.4f})"
        )
        print("---------------------------------------------------\n")
        # raise Exception(
        #     "DEBUG STOP AFTER PRINTING MASKS"
        # )  # 可以加這行強制停止，只看打印結果

        self._register_backward_hooks()
        _create_step_wrapper(self, optimizer)  # Modifies optimizer.step()

        # For storing L1/L2 gradients from external source
        self._grads_l1 = None
        self._grads_l2 = None

        print("RigLScheduler (Consistency Version) initialized.")
        print(self)  # Print initial state

    def _get_module_from_param_name(self, param_name):
        """Helper to get the module object a parameter belongs to."""
        parts = param_name.split(".")
        module = self.model
        for part in parts[
            :-1
        ]:  # Iterate until the last part (which is the param itself)
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None  # Should not happen if param_name is from model.named_parameters()
        return module

    def _init_sparsity_and_masks(self):
        """
        Initializes N, S, backward_masks, is_linear, and _param_names.
        This is a critical part. Aims for robust name-based initialization.
        """
        print(
            "Initializing RigLScheduler N, S, and mask structures (Corrected Bias/BatchNorm/Predictor Aware)..."
        )
        first_layer_candidate_name = None
        processed_param_names = []

        temp_params_info = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                module = self._get_module_from_param_name(name)
                is_linear = isinstance(module, torch.nn.Linear)
                # --- 判斷是否是偏置項 ---
                is_bias = name.endswith(".bias")
                # --- 判斷是否屬於 Predictor (根據你的模型結構調整) ---
                is_predictor_param = name.startswith("predictor.")
                # --- 判斷是否屬於 BatchNorm 層 (根據你的模型結構和命名調整，通常 BatchNorm 後面是 .weight 或 .bias) ---
                # 一個簡單的判斷，假設 BatchNorm 層名字裡包含 'bn' 或 'batchnorm'
                # 或者檢查模組類型
                is_batchnorm_param = isinstance(
                    module, torch.nn.BatchNorm2d
                ) or isinstance(
                    module, torch.nn.BatchNorm1d
                )  # 檢查模組類型更可靠

                temp_params_info.append(
                    {
                        "name": name,
                        "param": param,
                        "is_linear": is_linear,
                        "is_bias": is_bias,
                        "is_predictor_param": is_predictor_param,
                        "is_batchnorm_param": is_batchnorm_param,  # 新增這個標記
                        "numel": param.numel(),
                    }
                )
                processed_param_names.append(name)

        num_potentially_managed_layers = 0
        # 在判斷第一層時，要排除 bias, predictor 和 batchnorm
        for info in temp_params_info:
            if (
                not (info["is_linear"] and self.ignore_linear_layers)
                and not info["is_bias"]
                and not info["is_predictor_param"]
                and not info["is_batchnorm_param"]
            ):  # 新增排除 batchnorm
                num_potentially_managed_layers += 1
                if first_layer_candidate_name is None:
                    first_layer_candidate_name = info["name"]
                if num_potentially_managed_layers > 1:
                    break

        make_first_layer_dense_for_uniform = (
            self.sparsity_distribution == "uniform"
            and num_potentially_managed_layers > 1
            and first_layer_candidate_name is not None
        )

        for info in temp_params_info:
            name = info["name"]
            param = info["param"]
            is_linear_layer = info["is_linear"]
            is_bias_param = info["is_bias"]
            is_predictor_layer_param = info["is_predictor_param"]
            is_batchnorm_layer_param = info["is_batchnorm_param"]  # 獲取 batchnorm 標記

            self._param_names.append(name)
            self.N[name] = info["numel"]
            self.is_linear[name] = is_linear_layer

            current_sparsity_target = 0.0  # Default to dense

            # --- 核心修改：根據參數類型決定稀疏度 ---
            if is_bias_param:
                current_sparsity_target = 0.0  # 保持所有偏置項密集
            elif is_batchnorm_layer_param:
                current_sparsity_target = 0.0  # 保持所有 BatchNorm 參數密集 (新增)
            elif is_predictor_layer_param:
                current_sparsity_target = 0.0  # 保持 Predictor 層密集 (根據之前的決定)
            elif (
                make_first_layer_dense_for_uniform
                and name == first_layer_candidate_name
            ):
                current_sparsity_target = 0.0
            elif is_linear_layer and self.ignore_linear_layers:
                current_sparsity_target = 0.0
            else:
                current_sparsity_target = (
                    1.0 - self.dense_allocation
                )  # 其他層應用目標稀疏度

            self.S[name] = max(0.0, min(current_sparsity_target, 1.0 - 1e-9))

            if self.S[name] > 0:
                self.backward_masks[name] = torch.empty_like(param, dtype=torch.bool)
            else:
                self.backward_masks[name] = None

    @torch.no_grad()
    def random_sparsify_model(self):
        """Applies initial random sparsity to parameters based on self.S."""
        print(
            f"Randomly sparsifying model to dense_allocation={self.dense_allocation}..."
        )
        is_dist = dist.is_initialized()

        for name, param in self.model.named_parameters():
            if name not in self._param_names:  # Only operate on managed parameters
                continue

            target_sparsity = self.S.get(name, 0.0)
            if target_sparsity <= 0:  # Layer is dense
                self.backward_masks[name] = None
                # Ensure param data is not accidentally all zeros if it was dense
                # (though it shouldn't be if properly initialized)
                continue

            num_elements = self.N[name]
            num_to_prune = int(target_sparsity * num_elements)
            num_to_prune = min(max(0, num_to_prune), num_elements)  # Clamp

            # Create a random mask
            # True means keep, False means prune (mask is for *keeping* connections)
            initial_mask_flat = torch.ones(
                num_elements, device=param.device, dtype=torch.bool
            )
            if num_to_prune > 0:
                indices_to_prune = torch.randperm(num_elements, device=param.device)[
                    :num_to_prune
                ]
                initial_mask_flat[indices_to_prune] = False

            initial_mask = initial_mask_flat.reshape(param.shape)

            if is_dist:  # Synchronize mask across processes
                dist.broadcast(
                    initial_mask.float(), 0
                )  # Broadcast as float, then convert back
                initial_mask = initial_mask.bool()

            self.backward_masks[name] = initial_mask
            param.data.mul_(initial_mask)  # Apply mask to weights

        print("Initial model sparsification complete.")

    def _register_backward_hooks(self):
        """Registers backward hooks to parameters that will be sparsified."""
        self.hook_handles = {}  # Store hook handles for potential removal
        self.backward_hook_objects = {}

        for name, param in self.model.named_parameters():
            if name in self.S and self.S[name] > 0:  # Only hook sparsified layers
                if getattr(param, "_has_rigl_hook", False):
                    warnings.warn(
                        f"Parameter '{name}' already has a RigL hook.", RuntimeWarning
                    )
                    continue
                try:
                    hook_obj = IndexMaskHook(name, self)  # Name is correct here
                    handle = param.register_hook(hook_obj)
                    self.hook_handles[name] = handle
                    self.backward_hook_objects[name] = hook_obj
                    setattr(param, "_has_rigl_hook", True)
                except Exception as e:
                    warnings.warn(
                        f"Error registering hook for '{name}': {e}", RuntimeWarning
                    )

    # --- State Dict, String representation, Momentum/Weight Masking ---
    # (These are mostly similar to your version, ensure name-based logic is correct)
    # For brevity, I'll assume your existing state_dict, __str__, reset_momentum,
    # apply_mask_to_weights are largely okay if they use the name-based dicts correctly.
    # I will add apply_mask_to_gradients.

    def state_dict(self):
        cpu_masks = {
            name: m.cpu() if m is not None else None
            for name, m in self.backward_masks.items()
        }
        return {
            "dense_allocation": self.dense_allocation,
            "S": self.S,
            "N": self.N,
            "is_linear": self.is_linear,
            "_param_names": self._param_names,
            "hyperparams": {
                "T_end": self.T_end,
                "delta_T": self.delta_T,
                "alpha": self.alpha,
                "sparsity_distribution": self.sparsity_distribution,
                "ignore_linear_layers": self.ignore_linear_layers,
                "static_topo": self.static_topo,
                "consistency_lambda": self.consistency_lambda,
                "grad_accumulation_n": self.grad_accumulation_n,  # always 1
            },
            "step": self.step,
            "rigl_steps": self.rigl_steps,
            "backward_masks": cpu_masks,
        }

    def load_state_dict(self, state_dict):
        # Basic attributes
        self.dense_allocation = state_dict["dense_allocation"]
        self.S = state_dict["S"]
        self.N = state_dict["N"]
        self.is_linear = state_dict["is_linear"]
        self._param_names = state_dict["_param_names"]

        # Hyperparameters
        hp = state_dict["hyperparams"]
        self.T_end = hp["T_end"]
        self.delta_T = hp["delta_T"]
        self.alpha = hp["alpha"]
        self.sparsity_distribution = hp["sparsity_distribution"]
        self.ignore_linear_layers = hp["ignore_linear_layers"]
        self.static_topo = hp["static_topo"]
        self.consistency_lambda = hp["consistency_lambda"]
        self.grad_accumulation_n = hp.get(
            "grad_accumulation_n", 1
        )  # a default if missing

        self.step = state_dict["step"]
        self.rigl_steps = state_dict["rigl_steps"]

        # Load masks and move to device
        loaded_masks = state_dict["backward_masks"]
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
            warnings.warn(
                "Could not determine model device, loading masks to CPU.",
                RuntimeWarning,
            )

        self.backward_masks = {}
        current_param_shapes = {
            name: p.shape for name, p in self.model.named_parameters()
        }
        for name in self._param_names:
            if name in loaded_masks and loaded_masks[name] is not None:
                mask_tensor = loaded_masks[name]
                if (
                    name in current_param_shapes
                    and mask_tensor.shape == current_param_shapes[name]
                ):
                    self.backward_masks[name] = mask_tensor.to(device).bool()
                else:
                    warnings.warn(
                        f"Mask shape mismatch for {name} or param not found. Re-initializing mask for {name}.",
                        RuntimeWarning,
                    )
                    # Fallback: re-initialize mask based on S for this layer
                    # This requires S to be correctly loaded.
                    # For simplicity now, set to None (dense) or re-randomize if S indicates sparse.
                    # This part needs to be robust. Let's assume if S[name] > 0, re-random is safer.
                    if self.S.get(name, 0.0) > 0 and name in current_param_shapes:
                        warnings.warn(
                            f"Re-randomizing mask for {name} due to load error.",
                            RuntimeWarning,
                        )
                        # Simplified re-randomization for one layer
                        num_elements = self.N[name]
                        num_to_prune = int(self.S[name] * num_elements)
                        initial_mask_flat = torch.ones(
                            num_elements, device=device, dtype=torch.bool
                        )
                        if num_to_prune > 0:
                            indices_to_prune = torch.randperm(
                                num_elements, device=device
                            )[:num_to_prune]
                            initial_mask_flat[indices_to_prune] = False
                        self.backward_masks[name] = initial_mask_flat.reshape(
                            current_param_shapes[name]
                        )
                    else:
                        self.backward_masks[name] = None
            else:
                self.backward_masks[name] = None  # Layer was dense or mask missing

    @torch.no_grad()
    def reset_momentum(self):
        """Masks the momentum buffer in optimizer state."""
        if not hasattr(self.optimizer, "state") or not self.optimizer.state:
            return
        for name, param in self.model.named_parameters():
            if name in self.backward_masks and self.backward_masks[name] is not None:
                mask = self.backward_masks[name]
                if param in self.optimizer.state:
                    param_state = self.optimizer.state[param]
                    if "momentum_buffer" in param_state:
                        buf = param_state["momentum_buffer"]
                        if buf.shape == mask.shape:
                            buf.mul_(mask)
                        else:
                            warnings.warn(
                                f"Momentum buffer shape mismatch for {name}.",
                                RuntimeWarning,
                            )

    @torch.no_grad()
    def apply_mask_to_weights(self):
        """Ensures model weights conform to current masks."""
        for name, param in self.model.named_parameters():
            if name in self.backward_masks and self.backward_masks[name] is not None:
                mask = self.backward_masks[name]
                if param.shape == mask.shape:
                    param.data.mul_(mask)
                else:
                    warnings.warn(
                        f"Weight mask shape mismatch for {name}.", RuntimeWarning
                    )

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        """
        Zeros out gradients for connections that are currently pruned (mask is False).
        Original RigL calls this inside _rigl_step after mask update.
        """
        for name, param in self.model.named_parameters():
            if (
                param.grad is not None
                and name in self.backward_masks
                and self.backward_masks[name] is not None
            ):
                mask = self.backward_masks[name]
                if param.grad.shape == mask.shape:
                    param.grad.data.mul_(mask)
                else:
                    warnings.warn(
                        f"Gradient mask shape mismatch for {name}.", RuntimeWarning
                    )

    def __str__(self):
        # A more robust __str__
        s = f"RigLScheduler (Consistency, Name-Based)\n"
        s += f"  Step: {self.step}, RigL Steps: {self.rigl_steps}\n"
        s += f"  Dense Allocation: {self.dense_allocation:.2f} (Sparsity: {1-self.dense_allocation:.2f})\n"
        s += f"  Target Sparsity (S values per layer):\n"
        num_nonzero_total = 0
        num_elements_total = 0
        for name in self._param_names:
            s_val = self.S.get(name, 0.0)
            n_val = self.N.get(name, 0)
            mask = self.backward_masks.get(name)
            is_lin = self.is_linear.get(name, False)
            num_elements_total += n_val

            type_str = "Linear" if is_lin else "Conv/Other"
            if mask is not None:
                curr_nonzero = torch.sum(mask).item()
                num_nonzero_total += curr_nonzero
                s += f"    - {name} ({type_str}): S_target={s_val:.2f}, N={n_val}, Current Nonzero={curr_nonzero} ({curr_nonzero/max(1,n_val):.2%})\n"
            else:  # Dense
                num_nonzero_total += n_val  # All are non-zero
                s += f"    - {name} ({type_str}): S_target={s_val:.2f} (Dense), N={n_val}\n"
        s += f"  Overall Sparsity: Actual Nonzero = {num_nonzero_total}/{num_elements_total} ({num_nonzero_total/max(1,num_elements_total):.2%})\n"
        s += f"  Consistency Lambda: {self.consistency_lambda}\n"
        return s

    def check_if_backward_hook_should_accumulate_grad(self):
        # This logic is from original RigL, tied to grad_accumulation_n.
        # Since we force grad_accumulation_n = 1 and use w.grad directly,
        # this might not be strictly necessary for grow scores, but hooks might use it.
        if self.static_topo or self.step >= self.T_end:
            return False
        # Original logic: steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        # return steps_til_next_rigl_step <= self.grad_accumulation_n
        # With grad_accumulation_n = 1, it means only accumulate if next step is RigL step
        is_next_step_update = (self.step + 1) % self.delta_T == 0 and (
            self.step + 1
        ) < self.T_end
        return is_next_step_update

    def cosine_annealing(self):
        """Calculates drop fraction based on cosine annealing schedule."""
        if self.step >= self.T_end:
            return 0.0  # No more dropping/growing
        # Ensure T_end is not zero to avoid division by zero
        effective_T_end = max(self.T_end, 1)
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / effective_T_end))

    def set_additional_gradients(self, grads_l1, grads_l2):
        """Stores L1 and L2 gradients from the training loop."""
        if isinstance(grads_l1, dict) and isinstance(grads_l2, dict):
            self._grads_l1 = grads_l1
            self._grads_l2 = grads_l2
        else:
            warnings.warn(
                "L1/L2 gradients not set correctly (expected dicts).", RuntimeWarning
            )
            self._grads_l1 = None
            self._grads_l2 = None

    def __call__(self):
        """Called by the training loop, typically before optimizer.step()."""
        self.step += 1
        if self.static_topo:
            return True  # No RigL update, optimizer proceeds

        # Check if it's time for a RigL update
        is_update_time = (self.step % self.delta_T) == 0
        is_before_end_time = self.step < self.T_end

        if is_update_time and is_before_end_time:
            print(
                f"\n--- RigLScheduler: Performing update step {self.rigl_steps + 1} at global training step {self.step} ---"
            )
            self._rigl_step()
            self.rigl_steps += 1
            print(f"--- RigLScheduler: Update step {self.rigl_steps} finished. ---")
            # Clean up L1/L2 gradients after use
            self._grads_l1 = None
            self._grads_l2 = None
            return False  # Indicates RigL update occurred, optimizer might skip step if RigL did one (original RigL does not skip opt step)
            # The wrapper ensures RigL happens *after* optimizer step.
            # So, this return value True/False is more about signaling if an update happened.
        return True  # No RigL update occurred

    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()
        if drop_fraction <= 0:
            print("RigLScheduler: Drop fraction is zero or negative. Skipping update.")
            return

        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1

        # Determine if consistency logic should be used
        use_consistency = (
            self.consistency_lambda > 0  # Only if lambda is positive
            and isinstance(self._grads_l1, dict)
            and isinstance(self._grads_l2, dict)
        )
        if use_consistency:
            print("RigLScheduler: Using consistency-aware grow scores.")
        else:
            print("RigLScheduler: Using standard grow scores (abs(grad_total)).")

        for name in self._param_names:
            param = self.model.get_parameter(name)  # More robust way to get param
            current_mask = self.backward_masks.get(name)
            target_sparsity = self.S.get(name, 0.0)

            if (
                target_sparsity <= 0 or current_mask is None
            ):  # Skip dense or unmanaged layers
                continue

            grad_total = param.grad
            if grad_total is None:
                warnings.warn(
                    f"RigLScheduler: No gradient for '{name}'. Skipping parameter in _rigl_step.",
                    RuntimeWarning,
                )
                continue

            # --- Calculate Scores ---
            score_drop = torch.abs(param.data)  # Drop based on magnitude
            score_grow_total = torch.abs(grad_total)  # Base grow score

            if use_consistency:
                grad_l1 = self._grads_l1.get(name)  # type: ignore
                grad_l2 = self._grads_l2.get(name)  # type: ignore

                if grad_l1 is not None and grad_l2 is not None:
                    if grad_l1.shape == param.shape and grad_l2.shape == param.shape:
                        consistency_signal = torch.sign(grad_l1) * torch.sign(
                            grad_l2
                        )  # Element-wise
                        # Modifier: if grads align, factor > 1; if anti-align, factor < 1.
                        # Clamp to ensure positive and not excessively large/small.
                        # e.g., lambda=0.5 -> [0.5, 1.5]; lambda=1.0 -> [0.0, 2.0] (clamped to min=0.1)
                        consistency_modifier = torch.clamp(
                            1 + self.consistency_lambda * consistency_signal,
                            min=0.1,  # min=0.1 to avoid zero score
                        )
                        score_grow = score_grow_total * consistency_modifier
                    else:
                        warnings.warn(
                            f"L1/L2 grad shape mismatch for {name}. Using standard grow score.",
                            RuntimeWarning,
                        )
                        score_grow = score_grow_total
                else:
                    warnings.warn(
                        f"Missing L1 or L2 grad for {name}. Using standard grow score.",
                        RuntimeWarning,
                    )
                    score_grow = score_grow_total
            else:
                score_grow = score_grow_total  # Standard RigL grow score

            # --- Distributed Sync (if applicable) ---
            if is_dist:
                dist.all_reduce(score_drop, op=dist.ReduceOp.AVG)
                dist.all_reduce(score_grow, op=dist.ReduceOp.AVG)

            # --- Calculate number of connections to prune/keep ---
            num_elements = self.N[name]
            current_num_ones = torch.sum(current_mask).item()
            num_to_prune = int(current_num_ones * drop_fraction)
            # Ensure num_to_prune is not negative or greater than current_num_ones
            num_to_prune = min(max(0, num_to_prune), current_num_ones)
            num_to_keep = current_num_ones - num_to_prune

            # --- Create mask for connections to KEEP (mask1_flat) ---
            # Sort by score_drop (connections with small magnitude are dropped first)
            # We want to keep `num_to_keep` connections with the largest magnitude.
            # Original RigL sorts ascending and takes last `num_to_keep` or sorts descending and takes first `num_to_keep`.
            # Let's sort descending by score_drop and keep the top `num_to_keep`.
            score_drop_flat = score_drop.flatten()
            mask1_flat = torch.zeros_like(score_drop_flat, dtype=torch.bool)
            if num_to_keep > 0:
                # We need to select from *currently active* connections for dropping.
                # So, only consider scores where current_mask is True.
                active_scores_drop = score_drop_flat[current_mask.flatten()]
                if (
                    active_scores_drop.numel() > 0
                ):  # Ensure there are active connections
                    # Indices relative to the *flattened active* connections
                    _, sorted_indices_active_drop = torch.sort(
                        active_scores_drop, descending=True
                    )

                    # Get the original indices of these top `num_to_keep` active connections
                    original_indices_of_active = torch.where(current_mask.flatten())[0]
                    indices_to_keep_flat = original_indices_of_active[
                        sorted_indices_active_drop[:num_to_keep]
                    ]
                    mask1_flat[indices_to_keep_flat] = True
                elif (
                    num_to_keep > 0
                ):  # No active connections, but told to keep some (should not happen if current_num_ones was >0)
                    warnings.warn(
                        f"Logic error for {name}: num_to_keep > 0 but no active scores to drop from.",
                        RuntimeWarning,
                    )

            # --- Create mask for connections to GROW (mask2_flat) ---
            # Grow connections from those not in mask1_flat (i.e., dropped or were already zero)
            score_grow_flat = score_grow.flatten()
            # Penalize scores of connections already kept (in mask1_flat)
            # so they are not chosen for growth.
            grow_scores_penalized = score_grow_flat.clone()
            grow_scores_penalized[mask1_flat] = -float(
                "inf"
            )  # Effectively ignore already kept

            mask2_flat = torch.zeros_like(score_grow_flat, dtype=torch.bool)
            if num_to_prune > 0:  # num_to_prune is also num_to_grow
                # Sort penalized scores and pick top `num_to_prune` to grow
                _, sorted_indices_grow = torch.sort(
                    grow_scores_penalized, descending=True
                )
                indices_to_grow_flat = sorted_indices_grow[:num_to_prune]
                mask2_flat[indices_to_grow_flat] = True

            # --- Combine masks and update ---
            # mask_combined is the new desired state of active connections
            new_mask_topology = mask1_flat.reshape(param.shape) | mask2_flat.reshape(
                param.shape
            )

            # Identify connections that are *newly* grown
            # These are in new_mask_topology but were not in current_mask
            newly_grown_connections = new_mask_topology & (~current_mask)

            # Update parameter weights:
            # 1. Initialize newly grown connections to zero.
            param.data[newly_grown_connections] = 0.0
            # 2. Ensure all non-active connections (not in new_mask_topology) are zero.
            # This is handled by apply_mask_to_weights called by the wrapper,
            # but can also be done here for immediate effect if needed.
            # param.data.mul_(new_mask_topology) # Optional immediate application

            # Update the scheduler's mask for this parameter
            self.backward_masks[name] = new_mask_topology

        # After iterating all layers, apply masks to gradients
        # This step was in original RigL's _rigl_step
        self.apply_mask_to_gradients()
