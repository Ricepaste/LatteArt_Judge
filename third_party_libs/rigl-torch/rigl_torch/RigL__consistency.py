# third_party_libs/rigl-torch/rigl_torch/RigL__consistency.py (Refactored Version)

"""implementation of https://arxiv.org/abs/1911.11134"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import warnings  # For warnings instead of prints

from rigl_torch.util import (
    get_W,
    get_weighted_layers,
)  # Import get_weighted_layers as well


class IndexMaskHook:
    """
    Hook for RigL Scheduler. Stores parameter name instead of index.
    """

    def __init__(self, param_name, scheduler):
        self.param_name = param_name  # Store parameter name
        self.scheduler = scheduler
        # dense_grad accumulation is likely unused with external gradient calculation
        # but kept for potential future use or debugging
        self.dense_grad = None

    def __name__(self):
        return "IndexMaskHook"

    @torch.no_grad()
    def __call__(self, grad):
        # Access mask using parameter name
        mask = self.scheduler.backward_masks.get(self.param_name)

        # If no mask for this param (e.g., layer is dense), return original grad
        if mask is None:
            return grad

        # Check if mask shape matches grad shape
        if mask.shape != grad.shape:
            warnings.warn(
                f"Mask shape {mask.shape} mismatch with grad shape {grad.shape} in IndexMaskHook for '{self.param_name}'. Returning original grad.",
                RuntimeWarning,
            )
            return grad

        # Apply mask to gradient
        masked_grad = grad * mask

        # --- Optional: dense_grad accumulation (less relevant now) ---
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                self.dense_grad = torch.zeros_like(masked_grad)
            if self.dense_grad.shape == masked_grad.shape:
                if self.scheduler.grad_accumulation_n > 0:
                    self.dense_grad += masked_grad / self.scheduler.grad_accumulation_n
                else:
                    warnings.warn(
                        f"grad_accumulation_n is not positive ({self.scheduler.grad_accumulation_n}). Skipping dense_grad accumulation.",
                        RuntimeWarning,
                    )
            else:
                warnings.warn(
                    f"dense_grad shape mismatch in IndexMaskHook for '{self.param_name}'. Resetting dense_grad.",
                    RuntimeWarning,
                )
                self.dense_grad = torch.zeros_like(masked_grad)
        else:
            self.dense_grad = None
        # --- End optional dense_grad ---

        return masked_grad


def _create_step_wrapper(scheduler, optimizer):
    """Wraps optimizer.step() to include RigL logic."""
    _unwrapped_step = optimizer.step

    def _wrapped_step(*args, **kwargs):
        result = _unwrapped_step(*args, **kwargs)
        # Apply RigL maintenance after optimizer step
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()
        return result

    optimizer.step = _wrapped_step


class RigLScheduler:
    """
    RigL Scheduler implementation refactored to use parameter names as keys
    and incorporating consistency-based importance scores.
    """

    def __init__(
        self,
        model,
        optimizer,
        dense_allocation=1.0,
        T_end=1000,
        sparsity_distribution="uniform",
        ignore_linear_layers=True,  # This logic needs adjustment with name-based approach
        delta=100,
        alpha=0.3,
        consistency_lambda=0.5,
        static_topo=False,
        grad_accumulation_n=1,  # Forced to 1 internally
        state_dict=None,
    ):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise ValueError("Dense allocation must be on interval (0, 1].")
        if not isinstance(grad_accumulation_n, int) or grad_accumulation_n <= 0:
            warnings.warn(
                f"grad_accumulation_n invalid ({grad_accumulation_n}). Setting to 1.",
                RuntimeWarning,
            )
            grad_accumulation_n = 1

        self.model = model
        self.optimizer = optimizer
        self.dense_allocation = dense_allocation
        self.static_topo = static_topo
        self.consistency_lambda = consistency_lambda
        self.grad_accumulation_n = 1  # Force to 1
        self.sparsity_distribution = (
            sparsity_distribution  # Note: 'uniform' logic needs care
        )
        self.ignore_linear_layers = ignore_linear_layers  # Store for reference

        # --- Refactored State Initialization (using dictionaries) ---
        self.N = {}  # Dictionary for parameter counts {name: count}
        self.S = {}  # Dictionary for sparsity levels {name: sparsity_float}
        self.backward_masks = {}  # Dictionary for masks {name: mask_tensor or None}
        self.is_linear = {}  # Dictionary to track linear layers {name: bool}
        self.param_names = []  # List to store names of parameters managed by RigL

        # --- Populate state dictionaries using named_parameters ---
        # We still need to identify linear vs. non-linear, potentially first layer
        # Let's try a combined approach: iterate named_parameters, but use get_weighted_layers
        # just to get the layer type information if possible. This is still a bit fragile.
        # A better way is to analyze layer types directly during iteration.

        param_idx = 0
        first_layer_name = None
        total_rigl_managed_layers = 0

        print("Initializing RigLScheduler states...")
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Basic properties
                self.param_names.append(name)
                self.N[name] = param.numel()

                # --- Determine layer type and sparsity ---
                is_linear_layer = isinstance(
                    self._get_module_from_name(name), torch.nn.Linear
                )
                self.is_linear[name] = is_linear_layer

                # Find the first layer that requires grad and will be managed
                if first_layer_name is None and not (
                    is_linear_layer and self.ignore_linear_layers
                ):
                    first_layer_name = name

                sparsity_val = 0.0  # Default dense

                # Apply sparsity rules (uniform needs careful handling without strict order)
                # Let's define "first layer" as the first non-ignored layer encountered
                is_first_managed_layer = name == first_layer_name

                if (
                    is_first_managed_layer
                    and self.sparsity_distribution == "uniform"
                    # Check if there will be more than one managed layer eventually
                    # This is tricky to know upfront, maybe assume uniform applies if not linear ignored?
                    # Let's simplify: apply uniform only if not the very first layer overall?
                    # Or maybe apply (1-dense_allocation) to all non-ignored layers except first?
                    # Sticking to original logic attempt: make first *managed* layer dense
                ):
                    sparsity_val = 0.0
                elif is_linear_layer and self.ignore_linear_layers:
                    sparsity_val = 0.0  # Ignored linear layers are dense
                else:
                    sparsity_val = 1.0 - dense_allocation  # Apply target sparsity

                # Clamp and store sparsity
                sparsity_val = max(0.0, min(sparsity_val, 1.0 - 1e-9))
                self.S[name] = sparsity_val

                # Initialize mask dictionary entry
                if sparsity_val <= 0:
                    self.backward_masks[name] = None  # Dense
                else:
                    # Placeholder, actual mask created in random_sparsify
                    self.backward_masks[name] = torch.empty_like(
                        param, dtype=torch.bool
                    )
                    total_rigl_managed_layers += 1

                param_idx += 1

        # Adjust first layer density for uniform if only one layer managed
        if (
            self.sparsity_distribution == "uniform"
            and total_rigl_managed_layers == 1
            and first_layer_name
        ):
            print(
                f"Only one managed layer ('{first_layer_name}'). Applying target sparsity instead of keeping dense."
            )
            self.S[first_layer_name] = max(0.0, min(1.0 - dense_allocation, 1.0 - 1e-9))
            # Re-check if mask needs creation
            if (
                self.S[first_layer_name] > 0
                and self.backward_masks[first_layer_name] is None
            ):
                param = dict(self.model.named_parameters())[first_layer_name]
                self.backward_masks[first_layer_name] = torch.empty_like(
                    param, dtype=torch.bool
                )

        # --- End Refactored State Initialization ---

        if state_dict is not None:
            print("Loading RigLScheduler state...")
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()
            print("RigLScheduler state loaded.")
        else:
            # Initialize steps and schedule if not loading
            self.step = 0
            self.rigl_steps = 0
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end
            # Create initial random masks
            self.random_sparsify()

        # --- Refactored Hook Registration (using names) ---
        self.backward_hook_objects = {}  # Dict to store hooks {name: hook_obj}
        hook_handles = {}  # Dict to store hook handles {name: handle}

        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.S and self.S[name] > 0:
                if getattr(param, "_has_rigl_backward_hook", False):
                    warnings.warn(
                        f"Parameter '{name}' already has a RigL hook.", RuntimeWarning
                    )
                    continue
                try:
                    hook = IndexMaskHook(name, self)  # Pass name to hook
                    handle = param.register_hook(hook)
                    hook_handles[name] = handle
                    setattr(param, "_has_rigl_backward_hook", True)
                    self.backward_hook_objects[name] = hook
                except Exception as e:
                    warnings.warn(
                        f"Error registering hook for '{name}': {e}", RuntimeWarning
                    )
        # self.hook_handles = hook_handles # Optional: store handles if needed for removal
        # --- End Refactored Hook Registration ---

        # Initialize optimizer step wrapper
        _create_step_wrapper(self, optimizer)

        # Initialize external gradient holders
        self._grads_l1 = None
        self._grads_l2 = None

        print("RigLScheduler initialization complete.")
        print(self)  # Print status after init

    def _get_module_from_name(self, name):
        """Helper to get module object from parameter name."""
        mods = name.split(".")
        module = self.model
        for mod_name in mods[:-1]:  # Iterate through module path
            if hasattr(module, mod_name):
                module = getattr(module, mod_name)
            else:
                return None  # Module path invalid
        # The last part is the parameter name (e.g., 'weight', 'bias')
        # We want the module containing the parameter
        return module

    def state_dict(self):
        """Returns the state of the scheduler as a dictionary."""
        # Move masks to CPU for serialization
        cpu_masks = {}
        for name, mask in self.backward_masks.items():
            if isinstance(mask, torch.Tensor):
                cpu_masks[name] = mask.cpu()
            else:
                cpu_masks[name] = None  # Keep None as None

        obj = {
            "dense_allocation": self.dense_allocation,
            "S": self.S,  # Dict keyed by name
            "N": self.N,  # Dict keyed by name
            "is_linear": self.is_linear,  # Dict keyed by name
            "param_names": self.param_names,  # Store the list of managed names
            "hyperparams": {
                "delta_T": self.delta_T,
                "alpha": self.alpha,
                "T_end": self.T_end,
                "ignore_linear_layers": self.ignore_linear_layers,
                "static_topo": self.static_topo,
                "sparsity_distribution": self.sparsity_distribution,
                "grad_accumulation_n": self.grad_accumulation_n,  # Is always 1 now
                "consistency_lambda": self.consistency_lambda,
            },
            "step": self.step,
            "rigl_steps": self.rigl_steps,
            "backward_masks": cpu_masks,  # Dict keyed by name (CPU tensors)
            # _linear_layers_mask is replaced by is_linear dict
        }
        return obj

    def load_state_dict(self, state_dict):
        """Loads the scheduler state from a dictionary."""
        print("Loading state into RigLScheduler...")
        # Basic parameters
        self.dense_allocation = state_dict.get("dense_allocation", 1.0)
        self.step = state_dict.get("step", 0)
        self.rigl_steps = state_dict.get("rigl_steps", 0)
        self.param_names = state_dict.get("param_names", [])  # Load managed names

        # Load dictionaries (ensure keys match current model structure if possible)
        self.S = state_dict.get("S", {})
        self.N = state_dict.get("N", {})
        self.is_linear = state_dict.get("is_linear", {})

        # Load hyperparameters
        hyperparams = state_dict.get("hyperparams", {})
        self.delta_T = hyperparams.get("delta_T", 100)
        self.alpha = hyperparams.get("alpha", 0.3)
        self.T_end = hyperparams.get("T_end", 1000)
        self.ignore_linear_layers = hyperparams.get("ignore_linear_layers", True)
        self.static_topo = hyperparams.get("static_topo", False)
        self.sparsity_distribution = hyperparams.get("sparsity_distribution", "uniform")
        self.grad_accumulation_n = 1  # Force to 1
        self.consistency_lambda = hyperparams.get("consistency_lambda", 0.5)

        # Load masks (dictionary) and move to correct device
        loaded_masks_dict = state_dict.get("backward_masks", {})
        self.backward_masks = {}
        # Get device from current model parameters
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            warnings.warn(
                "Cannot determine model device, defaulting masks to CPU.",
                RuntimeWarning,
            )
            device = torch.device("cpu")

        current_params = dict(self.model.named_parameters())

        for name in self.param_names:  # Iterate through expected managed names
            if name not in current_params:
                warnings.warn(
                    f"Parameter '{name}' from loaded state not found in current model. Skipping mask loading.",
                    RuntimeWarning,
                )
                continue

            mask_state = loaded_masks_dict.get(name)
            target_shape = current_params[name].shape

            if mask_state is not None and isinstance(mask_state, torch.Tensor):
                if mask_state.shape == target_shape:
                    self.backward_masks[name] = mask_state.to(device).bool()
                else:
                    warnings.warn(
                        f"Shape mismatch for loaded mask '{name}'. Loaded: {mask_state.shape}, Target: {target_shape}. Discarding loaded mask.",
                        RuntimeWarning,
                    )
                    # Decide fallback: None (dense) or re-randomize? Let's use None.
                    self.backward_masks[name] = None
                    # Also update S dict if mask is discarded
                    if name in self.S:
                        self.S[name] = 0.0
            else:
                self.backward_masks[name] = None  # None remains None

        # Ensure all managed params have an entry in backward_masks
        for name in self.param_names:
            if name not in self.backward_masks:
                self.backward_masks[name] = None  # Assume dense if missing
                if name in self.S:
                    self.S[name] = 0.0

        print("State loading finished. Applying masks to weights.")
        # Apply mask to weights after loading state
        self.apply_mask_to_weights()

    @torch.no_grad()
    def random_sparsify(self):
        """Initializes masks with random sparsity using dictionary structure."""
        print("Randomly sparsifying model (name-based)...")
        is_dist = dist.is_initialized()
        current_params = dict(self.model.named_parameters())

        for name in self.param_names:
            if name not in current_params:
                warnings.warn(
                    f"Parameter '{name}' not found during random_sparsify.",
                    RuntimeWarning,
                )
                continue

            w = current_params[name]
            sparsity = self.S.get(name, 0.0)  # Get sparsity for this param
            n_total = self.N.get(name, w.numel())  # Get count or use current numel

            if w.numel() != n_total:
                warnings.warn(
                    f"N['{name}'] ({n_total}) mismatch w.numel() ({w.numel()}). Using w.numel().",
                    RuntimeWarning,
                )
                n_total = w.numel()
                self.N[name] = n_total  # Update N dict

            # Skip if dense
            if sparsity <= 0:
                self.backward_masks[name] = None
                continue

            n_sparse = int(sparsity * n_total)
            n_sparse = min(max(0, n_sparse), n_total)

            # Create and apply random mask
            perm = torch.randperm(n_total, device=w.device)
            perm = perm[:n_sparse]
            flat_mask = torch.ones(n_total, device=w.device)
            if perm.numel() > 0:
                flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask  # Apply initial mask to weights
            self.backward_masks[name] = mask  # Store the mask
        print("Model sparsified.")

    def __str__(self):
        """String representation using dictionary structure."""
        s = "RigLScheduler (name-based)(\n"
        num_managed_layers = len(self.param_names)
        s += f"managed_layers={num_managed_layers},\n"

        if num_managed_layers == 0:
            s += "  No parameters managed by RigL.\n"
            return s + ")"

        total_params = 0
        total_nonzero = 0
        total_conv_params = 0
        total_conv_nonzero = 0

        param_details = []

        current_params = dict(self.model.named_parameters())

        for name in self.param_names:
            if name not in current_params:
                continue  # Skip if param disappeared

            w = current_params[name]
            N = self.N.get(name, w.numel())
            S_val = self.S.get(name, 0.0)  # Sparsity value
            mask = self.backward_masks.get(name)
            is_lin = self.is_linear.get(name, False)

            if mask is None:  # Dense
                actual_nonzero = N
                percentage = 100.0
            elif mask.shape == w.shape:
                actual_nonzero = torch.sum(mask).item()  # Non-zero elements in mask
                percentage = float(actual_nonzero) / float(max(N, 1)) * 100
            else:  # Shape mismatch
                actual_nonzero = N
                percentage = 100.0
                param_details.append(
                    f"  '{name}': N={N}, S={S_val*100:.1f}%, Nonzero=N/A (Mask Shape Mismatch!)"
                )
                continue  # Skip detailed count for mismatch

            param_details.append(
                f"  '{name}': N={N}, S={S_val*100:.1f}%, Nonzero={actual_nonzero} ({percentage:.2f}%)"
            )

            total_params += N
            total_nonzero += actual_nonzero
            if not is_lin:
                total_conv_nonzero += actual_nonzero
                total_conv_params += N

        total_params = max(total_params, 1)
        total_conv_params = max(total_conv_params, 1)

        s += f"total_nonzero_params={total_nonzero}/{total_params} ({float(total_nonzero) / total_params * 100:.2f}%),\n"
        if total_conv_params > 0:
            s += f"total_CONV_nonzero_params={total_conv_nonzero}/{total_conv_params} ({float(total_conv_nonzero) / total_conv_params * 100:.2f}%),\n"

        s += "\n".join(param_details) + "\n"  # Add details for each layer
        s += f"step={self.step},\n"
        s += f"num_rigl_steps={self.rigl_steps},\n"
        s += f"ignoring_linear_layers={self.ignore_linear_layers},\n"
        s += f"sparsity_distribution={self.sparsity_distribution},\n"
        s += f"consistency_lambda={self.consistency_lambda}\n"

        return s + ")"

    @torch.no_grad()
    def reset_momentum(self):
        """Masks the momentum buffer using name-based masks."""
        if not self.optimizer.state:
            return

        current_params = dict(self.model.named_parameters())

        for name, param in current_params.items():
            if name not in self.param_names or name not in self.backward_masks:
                continue  # Only process managed parameters with masks

            mask = self.backward_masks[name]
            s = self.S.get(name, 0.0)

            if s <= 0 or mask is None:
                continue  # Skip dense

            if param not in self.optimizer.state:
                continue  # Param might not be in this opt group

            param_state = self.optimizer.state[param]
            if "momentum_buffer" in param_state:
                buf = param_state["momentum_buffer"]
                if buf.shape == mask.shape:
                    buf *= mask
                else:
                    warnings.warn(
                        f"Shape mismatch in reset_momentum for '{name}'.",
                        RuntimeWarning,
                    )

    @torch.no_grad()
    def apply_mask_to_weights(self):
        """Applies the current masks to the model weights using names."""
        current_params = dict(self.model.named_parameters())

        for name, param in current_params.items():
            if name not in self.param_names or name not in self.backward_masks:
                continue  # Only process managed parameters with masks

            mask = self.backward_masks[name]
            s = self.S.get(name, 0.0)

            if s <= 0 or mask is None:
                continue  # Skip dense

            if param.shape == mask.shape:
                param *= mask
            else:
                warnings.warn(
                    f"Shape mismatch in apply_mask_to_weights for '{name}'.",
                    RuntimeWarning,
                )

    def check_if_backward_hook_should_accumulate_grad(self):
        """Signals if external gradient calculation might be needed soon."""
        if self.static_topo or self.step >= self.T_end:
            return False
        # Signal if the *next* step is a RigL update step
        is_next_step_update = (self.step + 1) % self.delta_T == 0 and (
            self.step + 1
        ) < self.T_end
        return is_next_step_update

    def set_additional_gradients(self, grads_l1, grads_l2):
        """Stores the externally computed L1 and L2 gradients (dictionaries)."""
        if isinstance(grads_l1, dict) and isinstance(grads_l2, dict):
            self._grads_l1 = grads_l1
            self._grads_l2 = grads_l2
        else:
            warnings.warn(
                "set_additional_gradients received non-dict inputs. Gradients not set.",
                RuntimeWarning,
            )
            self._grads_l1 = None
            self._grads_l2 = None

    def cosine_annealing(self):
        """Calculates the drop fraction using cosine annealing."""
        if self.step >= self.T_end:
            return 0.0
        T_end_eff = max(self.T_end, 1)
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / T_end_eff))

    def __call__(self):
        """Performs the RigL update step if scheduled."""
        self.step += 1
        if self.static_topo:
            return True

        is_update_time = (self.step % self.delta_T) == 0
        is_before_end = self.step < self.T_end

        if is_update_time and is_before_end:
            if not isinstance(self._grads_l1, dict) or not isinstance(
                self._grads_l2, dict
            ):
                warnings.warn(
                    f"L1/L2 gradients not set before RigL step {self.step}. Skipping update.",
                    RuntimeWarning,
                )
                self._grads_l1 = self._grads_l2 = None  # Clear just in case
                return True  # No update occurred

            print(
                f"\nPerforming RigL Step {self.rigl_steps + 1} at global step {self.step}"
            )
            self._rigl_step()  # Execute the update logic
            self.rigl_steps += 1
            print(f"RigL Step {self.rigl_steps} finished.\n")

            # Clean up stored gradients
            self._grads_l1 = None
            self._grads_l2 = None
            return False  # RigL update occurred
        return True  # No RigL update

    @torch.no_grad()
    def _rigl_step(self):
        """The core RigL update logic (name-based)."""
        drop_fraction = self.cosine_annealing()
        if drop_fraction <= 0:
            print("Drop fraction is zero, skipping RigL step.")
            return

        # --- Gradient Dictionary Check ---
        if not isinstance(self._grads_l1, dict) or not isinstance(self._grads_l2, dict):
            print(
                "Error: _rigl_step called without valid gradient dictionaries. Aborting."
            )
            return

        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else 1
        current_params = dict(self.model.named_parameters())

        # --- Iterate through managed layers by name ---
        for name in self.param_names:
            # --- Get current parameter, sparsity, mask ---
            if name not in current_params:
                continue  # Param might have been removed
            w = current_params[name]
            sparsity = self.S.get(name, 0.0)
            current_mask = self.backward_masks.get(name)

            # Skip dense layers or layers without a mask managed by RigL
            if sparsity <= 0 or current_mask is None:
                continue

            # --- Get Gradients ---
            grad_l1 = self._grads_l1.get(name)
            grad_l2 = self._grads_l2.get(name)
            grad_total = w.grad  # Assumes L_total.backward() was just called

            # --- Sanity Checks ---
            if grad_l1 is None or grad_l2 is None or grad_total is None:
                warnings.warn(
                    f"Missing gradients for '{name}'. Skipping.", RuntimeWarning
                )
                continue
            if (
                current_mask.shape != w.shape
                or grad_l1.shape != w.shape
                or grad_l2.shape != w.shape
                or grad_total.shape != w.shape
            ):
                warnings.warn(f"Shape mismatch for '{name}'. Skipping.", RuntimeWarning)
                continue

            # --- Calculate Scores ---
            score_drop = torch.abs(w)
            score_grow_total = torch.abs(grad_total)
            # consistency_factor_elementwise = torch.sign(grad_l1) * torch.sign(grad_l2)
            # consistency_modifier = torch.clamp(
            #     1 + self.consistency_lambda * consistency_factor_elementwise, min=0.1
            # )
            # # 把 grow_factor 限制在一個比較合理的正數範圍內，例如 [0.1, 1.5]
            # # 下限 0.1 (避免太小)，上限 1.5 (避免 lambda 太大時增長過度，可以調整)
            # score_grow = score_grow_total * consistency_modifier

            score_grow = score_grow_total  # 直接使用原始的 score_grow_total

            # --- Distributed Sync ---
            if is_dist:
                dist.all_reduce(score_drop, op=dist.ReduceOp.AVG)
                dist.all_reduce(score_grow, op=dist.ReduceOp.AVG)

            # --- Calculate prune/keep numbers ---
            n_total = w.numel()
            n_ones = torch.sum(current_mask).item()
            n_ones = min(max(0, int(n_ones)), n_total)
            n_prune = int(n_ones * drop_fraction)
            n_prune = min(max(0, n_prune), n_ones)
            n_keep = n_ones - n_prune

            # --- Create Drop Mask ---
            score_drop_flat = score_drop.view(-1)
            _, sorted_indices_drop = torch.sort(score_drop_flat, descending=True)
            mask1_flat = torch.zeros_like(score_drop_flat, dtype=torch.bool)
            if n_keep > 0:  # Handle edge case n_keep = 0
                mask1_flat[sorted_indices_drop[:n_keep]] = True

            # --- Create Grow Mask ---
            score_grow_flat = score_grow.view(-1)
            min_score = torch.min(score_grow_flat) - 1.0
            # Use mask1_flat (kept connections) to exclude from grow candidates
            score_grow_lifted = torch.where(mask1_flat, min_score, score_grow_flat)
            _, sorted_indices_grow = torch.sort(score_grow_lifted, descending=True)
            mask2_flat = torch.zeros_like(score_grow_flat, dtype=torch.bool)
            if n_prune > 0:  # Handle edge case n_prune = 0
                mask2_flat[sorted_indices_grow[:n_prune]] = True

            # --- Combine Masks and Update Weights ---
            mask1 = torch.reshape(mask1_flat, w.shape)
            mask2 = torch.reshape(mask2_flat, w.shape)

            # Final mask includes kept connections and newly grown connections
            mask_combined = mask1 | mask2

            # Identify *only* the newly grown connections to initialize them
            # These are connections present in mask2 but not in the *original* mask
            new_connections = mask2 & (~current_mask)

            # Update weights: Set newly grown connections to zero.
            # Existing connections within the combined mask retain their values.
            w.data[new_connections] = 0.0

            # Update the stored backward mask for the next iteration's hook
            self.backward_masks[name] = mask_combined

        # Momentum reset and weight masking are handled by the optimizer step wrapper
