import torch
import torch.nn as nn
import torch.nn.functional as F
import contextlib
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from typing import Tuple
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss

import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import _LRScheduler


class GroupAwarePolyLRScheduler(_LRScheduler):
    def __init__(self, optimizer, max_steps: int, exponent: float = 0.9):
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        decay_factor = (1 - current_step / self.max_steps) ** self.exponent

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * decay_factor


class DC_and_SMP_Focal_loss(nn.Module):
    def __init__(self, focal_kwargs, dice_kwargs, weight_focal=0.5, weight_dice=1.0):
        super(DC_and_SMP_Focal_loss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.dice = MemoryEfficientSoftDiceLoss(**dice_kwargs)
        self.focal = smp.losses.FocalLoss(**focal_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dice(net_output, target)
        if target.dim() == net_output.dim():
            focal_target = target.squeeze(1)
        else:
            focal_target = target
        focal_loss = self.focal(net_output, focal_target)
        result = (self.weight_dice * dc_loss) + (self.weight_focal * focal_loss)
        return result


class DynamicPromptModule3D(nn.Module):
    def __init__(self, in_channels, num_prompts=10, prompt_dim=256, top_k=10, temperature=0.09):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.top_k = top_k
        self.temperature = temperature

        self.prompts = nn.Parameter(torch.randn(num_prompts, prompt_dim))
        self.pool = nn.AdaptiveAvgPool3d((2, 2, 2))

        pooled_dim = in_channels * 8

        self.selector = nn.Sequential(
            nn.Linear(pooled_dim, pooled_dim // 2),
            nn.ReLU(),
            nn.Linear(pooled_dim // 2, num_prompts)
        )

        self.gamma_proj = nn.Linear(prompt_dim, in_channels)
        self.beta_proj = nn.Linear(prompt_dim, in_channels)



        self.register_buffer('running_weight_sum', torch.zeros(num_prompts))
        self.register_buffer('running_count', torch.tensor(0.0))



    def forward(self, x):
        B, C, D, H, W = x.shape

        z = self.pool(x).view(B, -1)
        logits = self.selector(z)
        #print(logits)

        topk_values, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        masked_logits = torch.full_like(logits, float('-inf'))
        masked_logits.scatter_(-1, topk_indices, topk_values)

        weights = F.softmax(masked_logits / self.temperature, dim=-1)
        # print("Prompt module called")
        # print(self.stage_name)
        # print(self.predictor is None)
        # print(self.predictor.current_case)
        # print(self.predictor.selected_cases)
        if hasattr(self, "predictor"):
            pred = self.predictor

            if pred.selected_cases is not None:

                if pred.current_case in pred.selected_cases:
                    stage = self.stage_name

                    w = weights.detach().cpu().numpy().squeeze()

                    if pred.current_case not in pred.prompt_logs[stage]:
                        pred.prompt_logs[stage][pred.current_case] = []

                    pred.prompt_logs[stage][pred.current_case].append(w)
                    #
                    # pred.prompt_logs[stage][pred.current_case] = \
                    #     weights.detach().cpu().numpy().squeeze()
        print("Saving weights")
        # save latest weights for debugging
        self.latest_prompt_weights = weights.detach().cpu()
        selected_prompt = weights @ self.prompts
       # print(weights)

        raw_gamma = self.gamma_proj(selected_prompt).view(B, C, 1, 1, 1)
        raw_beta = self.beta_proj(selected_prompt).view(B, C, 1, 1, 1)

        gamma = torch.tanh(raw_gamma)
        beta = torch.tanh(raw_beta)

        #print(beta.abs().mean(), beta.mean(), beta.std(), raw_beta.mean(), raw_beta.std())
       # print(gamma.abs().mean(), gamma.mean(), gamma.std(), raw_gamma.mean(), raw_gamma.std())
        #print("#############################################################################################")

        x_modulated = x * (1 + gamma) + beta

        if self.training:
            with torch.no_grad():
                self.running_weight_sum += weights.sum(dim=0)
                self.running_count += weights.shape[0]

        return x_modulated, weights

    def get_and_reset_epoch_stats(self):
        if self.running_count == 0:
            return None
        mean_weights = self.running_weight_sum / self.running_count
        self.running_weight_sum.zero_()
        self.running_count.zero_()
        return mean_weights


def patch_unet_forward_with_prompts(network: PlainConvUNet, num_stages_to_modulate: int = 3):
    """
    MODIFIED: Now injects a separate DynamicPromptModule at the deepest N stages.
    """
    out_channels = network.encoder.output_channels
    total_stages = len(out_channels)

    # 1. Create a dictionary to hold distinct modules for the deepest stages
    network.dynamic_prompt_modules = nn.ModuleDict()
    network.last_prompt_weights = {}  # Will store weights per stage

    # Determine which stages to modulate (e.g., if total=6, and num=3, stages are 3, 4, 5)
    start_stage = max(0, total_stages - num_stages_to_modulate)

    for i in range(start_stage, total_stages):
        module = DynamicPromptModule3D(in_channels=out_channels[i])

        module.stage_name = f"stage{i - start_stage + 1}"
        module.predictor = None

        network.dynamic_prompt_modules[str(i)] = module

    # Define the new multi-stage forward pass
    def new_forward(self, x):
        # Run nnU-Net Encoder
        skips = self.encoder(x)
        self.last_prompt_weights = {}

        # 2. Iterate through all skip connections and apply modulation IF a module exists for it
        for i in range(len(skips)):
            stage_key = str(i)
            if stage_key in self.dynamic_prompt_modules:
                prompted_skip, weights = self.dynamic_prompt_modules[stage_key](skips[i])
                skips[i] = prompted_skip
                self.last_prompt_weights[stage_key] = weights

        # Run nnU-Net Decoder
        return self.decoder(skips)

    import types
    network.forward = types.MethodType(new_forward, network)
    return network

def count_parameter_groups(model):
    total = sum(p.numel() for p in model.parameters())

    prompt = 0
    selector = 0
    gamma_beta = 0
    backbone = 0

    for name, p in model.named_parameters():
        if "dynamic_prompt_modules" in name:
            if "selector" in name:
                selector += p.numel()
            elif "gamma_proj" in name or "beta_proj" in name:
                gamma_beta += p.numel()
            else:
                prompt += p.numel()
        else:
            backbone += p.numel()

    print("=" * 70)
    print(f"Backbone:             {backbone:,}")
    print(f"Prompt vectors:       {prompt:,}")
    print(f"Selector networks:    {selector:,}")
    print(f"Gamma/Beta projectors:{gamma_beta:,}")
    print("-" * 70)
    print(f"Total:                {total:,}")
    print("=" * 70)

class nnUNetTrainer_DynamicMultiPrompts(nnUNetTrainer):

    def load_checkpoint(self, filename_or_checkpoint, *args, **kwargs):
        # 1. Load the model, optimizer, and scheduler states from the file
        super().load_checkpoint(filename_or_checkpoint, *args, **kwargs)

        # 2. Define your new base learning rates for each group
        # Index 0: Base params, Index 1: Prompt params, Index 2: Selector params
        new_lrs = [
            self.initial_lr * 0.01,  # Base network (1e-4)
            self.initial_lr * 0.1,  # Prompts (1e-3)
            self.initial_lr * 0.01  # Selector (1e-4)
        ]

        # 3. Override the optimizer's CURRENT learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = new_lrs[i]

            # If your optimizer uses 'initial_lr' internally (some PyTorch versions do)
            if 'initial_lr' in group:
                group['initial_lr'] = new_lrs[i]

        # 4. OVERRIDE THE SCHEDULER'S BASE LRs! (Crucial step)
        # This prevents the scheduler from reverting to the old checkpoint values
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            if hasattr(self.lr_scheduler, 'base_lrs'):
                self.lr_scheduler.base_lrs = new_lrs

        print(f"DEBUG: Optimizer and Scheduler base learning rates overridden to: {new_lrs}")

    def configure_optimizers(self):
        prompt_params = []
        selector_params = []
        base_params = []

        # FIX A: Use self.network instead of model
        for name, param in self.network.named_parameters():
            if "dynamic_prompt" in name:
                if "selector" in name:
                    selector_params.append(param)
                else:
                    prompt_params.append(param)
            else:
                base_params.append(param)

        # FIX B: Use nnU-Net's internal initial_lr and weight_decay
        # We assign the selector a learning rate that is 10x smaller than the base LR
        base_lr = self.initial_lr*0.01 # Usually 0.01 in nnU-Net
        selector_lr = self.initial_lr * 0.01 # This becomes 0.001
        prompt_lr = self.initial_lr*0.1

        optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': base_lr},
            {'params': prompt_params, 'lr': prompt_lr},
            {'params': selector_params, 'lr': selector_lr}
        ], momentum=0.99, nesterov=True, weight_decay=self.weight_decay) # Added weight_decay back!

        from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
        # Using your custom scheduler
        lr_scheduler = GroupAwarePolyLRScheduler(optimizer, self.num_epochs)
        return optimizer, lr_scheduler


    def build_loss(self):
        dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': False,
            'ddp': self.is_ddp
        }
        focal_kwargs = {
            'mode': 'multiclass',
            'gamma': 2.0,
            'alpha': None,
            'reduction': 'mean'
        }
        loss = DC_and_SMP_Focal_loss(
            focal_kwargs=focal_kwargs,
            dice_kwargs=dice_kwargs,
            weight_dice=0.5,
            weight_focal=0.5
        )
        return loss

    @staticmethod
    def build_network_architecture(plans_manager,
                                   dataset_json,
                                   configuration_manager,
                                   num_input_channels,
                                   enable_deep_supervision: bool = True):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
        network = nnUNetTrainer.build_network_architecture(
            plans_manager,
            dataset_json,
            configuration_manager,
            num_input_channels,
            enable_deep_supervision
        )

        # Apply multi-stage patching (defaults to modulating the deepest 3 stages)
        network = patch_unet_forward_with_prompts(network, num_stages_to_modulate=3)
        count_parameter_groups(network)
        return network

    def compute_orthogonality_loss(self, prompts):
        normed_prompts = F.normalize(prompts, p=2, dim=1)
        sim_matrix = torch.matmul(normed_prompts, normed_prompts.T)
        identity = torch.eye(sim_matrix.size(0)).to(sim_matrix.device)
        loss = F.mse_loss(sim_matrix, identity)
        return loss

    def train_step(self, batch: dict) -> dict:
        data = batch['data'].to(self.device, non_blocking=True)
        target = batch['target']
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)

        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else contextlib.nullcontext():
            output = self.network(data)

            l_dice_ce = self.loss(output, target)

            net_ref = self.network.module if hasattr(self.network, 'module') else self.network

            # MODIFIED: Calculate Orthogonality Loss across ALL active modules and average it
            l_ortho = 0.0
            prompt_dict = net_ref.dynamic_prompt_modules

            for stage_idx, module in prompt_dict.items():
                l_ortho += self.compute_orthogonality_loss(module.prompts)

            # Average the orthogonality loss over however many modules we have
            l_ortho = l_ortho / len(prompt_dict)

            l = l_dice_ce + 0.01 * l_ortho

        self.grad_scaler.scale(l).backward()
        self.grad_scaler.unscale_(self.optimizer)

        # MODIFIED: Name check updated to "dynamic_prompt"
        unet_params = [p for n, p in self.network.named_parameters() if 'dynamic_prompt' not in n]
        prompt_params = [p for n, p in self.network.named_parameters() if 'dynamic_prompt' in n]

        torch.nn.utils.clip_grad_norm_(unet_params, 12.0)
        if len(prompt_params) > 0:
            torch.nn.utils.clip_grad_norm_(prompt_params, 1.0)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return {'loss': l.detach().cpu().numpy()}

    def on_train_epoch_end(self, *args, **kwargs):
        super().on_train_epoch_end(*args, **kwargs)

        net_ref = self.network.module if hasattr(self.network, 'module') else self.network

        # MODIFIED: Iterate over the ModuleDict and print stats for EACH stage cleanly
        if hasattr(net_ref, 'dynamic_prompt_modules'):
            log_str = f"Epoch {self.current_epoch} | Mean Prompt Probabilities:\n"

            for stage_idx, module in net_ref.dynamic_prompt_modules.items():
                mean_weights = module.get_and_reset_epoch_stats()

                if mean_weights is not None:
                    weights_np = mean_weights.cpu().numpy().round(4)
                    log_str += f"  Stage {stage_idx}: [{', '.join(map(str, weights_np))}]\n"

            self.print_to_log_file(log_str.strip())
