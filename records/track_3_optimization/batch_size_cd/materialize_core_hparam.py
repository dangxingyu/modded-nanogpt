#!/usr/bin/env python3
"""Materialize core Track 3 hyperball optimizer recipes for hparam CD."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

RECIPES = {
    "muonh_core": {
        "path": "records/track_3_optimization/results/20260430_muonh/9319c798-6643-464a-b407-b05468e468f5.txt",
        "logfile": True,
    },
    "adamh_core": {
        "path": "records/track_3_optimization/results/20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac.txt",
        "logfile": True,
    },
    "lionh_core": {
        "path": "records/track_3_optimization/results/20260430_adamh/7533dd87-107f-4a4f-8229-acbec0fb00ac.txt",
        "logfile": True,
        "transform": "lionh_from_adamh",
    },
    "klsoap_h_core": {
        "path": "records/track_3_optimization/results/20260508_klsoap_h_clean_tuple_sweep/b1095_sh090/klsoap-h-b1095_sh090-K3125-seed-1.full.txt",
        "logfile": True,
    },
    "shampooh_core": {
        "path": "records/track_3_optimization/results/sh-origpinv-s3375-lr1em2-wd010-b9em1-ge15-pf1-near1-record-130563a2-b40b-43c1-8fb4-b7a3bbfa5969.txt",
        "logfile": True,
        "uses_shampoo": True,
        "transform": "shampooh_from_rohan",
    },
    "psgdh_core": {
        "path": "records/track_3_optimization/results/20260527_psgd/train_psgd.py",
        "transform": "psgdh_from_pr316",
    },
    "psgdh_pr316_exact": {
        "path": "records/track_3_optimization/results/20260527_psgd/train_psgd.py",
        "exact_passthrough": True,
    },
}


SHAMPOO_VENDOR = "records/track_3_optimization/results/20260513_shampoo_1_4_power"


LIONH_CLASS = r'''
class LionH(torch.optim.Optimizer):
    """Lion update direction applied through the same Frobenius-norm-preserving
    hyperball projection used by AdamH/MuonH."""
    def __init__(self, params, lr=0.005, betas=(0.9, 0.95), eps=1e-10):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                update = exp_avg.mul(beta1).add(p.grad, alpha=1 - beta1).sign()
                exp_avg.mul_(beta2).add_(p.grad, alpha=1 - beta2)
                scale_invariant_update_(p, update, group["lr"], eps=eps)
'''


SHAMPOOH_CLASS = r'''
class ShampooH(DistributedShampoo):
    """DistributedShampoo direction followed by a per-matrix hyperball projection.

    Rohan Anil's Shampoo code already distributes full 2D matrix blocks across ranks when
    max_preconditioner_dim is larger than the model matrices. We reuse its search direction
    computation and communication path, but replace additive parameter updates with the
    norm-preserving hyperball target used by the H-family baselines.
    """
    @torch.no_grad()
    def _per_group_step_impl(
        self,
        state_lists,
        step,
        lr,
        beta1,
        beta3,
        weight_decay,
        peak_lr,
        weight_decay_type,
        grafting_config_not_none,
        perform_amortized_computation,
        use_bias_correction,
        use_grafting_method,
        train_interp_coeff,
        eval_interp_coeff,
    ) -> None:
        unit_lr = torch.ones_like(lr)
        zero_weight_decay = torch.zeros_like(weight_decay) if torch.is_tensor(weight_decay) else 0.0
        blocked_search_directions = self._compute_search_directions(
            state_lists=state_lists,
            step=step,
            lr=unit_lr,
            beta1=beta1,
            beta3=beta3,
            weight_decay=zero_weight_decay,
            peak_lr=peak_lr,
            weight_decay_type=weight_decay_type,
            grafting_config_not_none=grafting_config_not_none,
            perform_amortized_computation=perform_amortized_computation,
            use_bias_correction=use_bias_correction,
            use_grafting_method=use_grafting_method,
            train_interp_coeff=train_interp_coeff,
            eval_interp_coeff=eval_interp_coeff,
        )
        local_params = state_lists["distributor"].local_masked_blocked_params
        lr_value = float(lr.item())
        wd_value = float(weight_decay.item()) if hasattr(weight_decay, "item") else float(weight_decay)
        hyperball_deltas = []
        for param, additive_direction in zip(local_params, blocked_search_directions, strict=True):
            update = additive_direction.neg()
            p_norm = param.norm()
            u_norm = update.norm()
            new_param = param - lr_value * update * p_norm / torch.clamp(u_norm, min=1e-10)
            new_param = new_param / torch.clamp(new_param.norm(), min=1e-10) * p_norm
            if wd_value:
                new_param = new_param * max(0.0, 1.0 - lr_value * wd_value)
            hyperball_deltas.append(new_param - param)
        state_lists["distributor"].update_params(tuple(hyperball_deltas))
'''


INJECT_HELPER = r'''

def _track3_cd_float_env(name, default):
    return float(os.environ.get(name, str(default)))

def _track3_cd_apply_decoupled_weight_decay(param, group):
    weight_decay = float(group.get("weight_decay", 0.0) or 0.0)
    if weight_decay:
        param.mul_(max(0.0, 1.0 - float(group["lr"]) * weight_decay))

def _track3_cd_weight_decay_eta(step, train_steps):
    warmup_frac = max(0.0, _track3_cd_float_env("TRACK3_WD_WARMUP_FRAC", 0.0))
    if warmup_frac <= 0:
        return 1.0
    progress = step / train_steps
    return min(1.0, progress / warmup_frac)

def _track3_cd_set_scheduled_weight_decay(group, step, train_steps):
    if "initial_weight_decay" in group:
        group["weight_decay"] = group["initial_weight_decay"] * _track3_cd_weight_decay_eta(step, train_steps)

def _track3_cd_clamp_beta(beta):
    return min(0.9999, max(0.0, beta))

def _track3_cd_keep_beta1(beta, om_mult):
    return _track3_cd_clamp_beta(1.0 - (1.0 - beta) * om_mult)

def _track3_cd_timescale_beta(beta, om_mult, batch_ratio):
    centered = beta ** batch_ratio
    return _track3_cd_clamp_beta(1.0 - (1.0 - centered) * om_mult)

def _track3_cd_apply_scaled_hparams(optimizers):
    recipe = os.environ.get("TRACK3_RECIPE", "")
    batch_ratio = _track3_cd_float_env("TRACK3_BATCH_RATIO", batch_size / (8 * 64 * 1024))
    lr_scale = _track3_cd_float_env("TRACK3_BASE_LR_SCALE", batch_ratio ** 0.5)
    matrix_lr_mult = _track3_cd_float_env("TRACK3_MATRIX_LR_MULT", 1.0)
    aux_lr_mult = _track3_cd_float_env("TRACK3_AUX_LR_MULT", 1.0)
    matrix_wd_peak = _track3_cd_float_env("TRACK3_MATRIX_WD_PEAK", 0.0)
    aux_wd_peak = _track3_cd_float_env("TRACK3_AUX_WD_PEAK", 0.0)
    matrix_beta1_om_mult = _track3_cd_float_env("TRACK3_MATRIX_BETA1_OM_MULT", 1.0)
    matrix_beta2_om_mult = _track3_cd_float_env("TRACK3_MATRIX_BETA2_OM_MULT", 1.0)
    precond_lr_mult = _track3_cd_float_env("TRACK3_PRECOND_LR_MULT", 1.0)
    shampoo_beta_om_mult = _track3_cd_float_env("TRACK3_SHAMPOO_BETA_OM_MULT", 1.0)
    aux_beta1_om_mult = _track3_cd_float_env("TRACK3_AUX_BETA1_OM_MULT", 1.0)
    aux_beta2_om_mult = _track3_cd_float_env("TRACK3_AUX_BETA2_OM_MULT", 1.0)
    skip_matrix_group_beta_scaling = recipe in {"shampooh_core"}

    for opt_idx, opt in enumerate(optimizers):
        is_aux = opt_idx == 0
        role_lr_mult = aux_lr_mult if is_aux else matrix_lr_mult
        role_wd_peak = aux_wd_peak if is_aux else matrix_wd_peak
        beta1_om_mult = aux_beta1_om_mult if is_aux else matrix_beta1_om_mult
        beta2_om_mult = aux_beta2_om_mult if is_aux else matrix_beta2_om_mult
        for group in opt.param_groups:
            if "initial_lr" in group:
                group["lr"] = group["initial_lr"] * lr_scale * role_lr_mult
                group["initial_lr"] = group["lr"]
            elif "lr" in group:
                group["lr"] = group["lr"] * lr_scale * role_lr_mult
            group["initial_weight_decay"] = role_wd_peak
            group["weight_decay"] = role_wd_peak
            skip_betas = skip_matrix_group_beta_scaling and not is_aux
            if "betas" in group and not skip_betas:
                b1, b2 = group["betas"]
                group["betas"] = (
                    _track3_cd_keep_beta1(b1, beta1_om_mult),
                    _track3_cd_timescale_beta(b2, beta2_om_mult, batch_ratio),
                )
            for key in ("beta1", "mu", "momentum"):
                if key in group and not skip_betas:
                    group[key] = _track3_cd_keep_beta1(group[key], beta1_om_mult)
            if "beta2" in group and not skip_betas:
                group["beta2"] = _track3_cd_timescale_beta(group["beta2"], beta2_om_mult, batch_ratio)
            if "shampoo_beta" in group:
                group["shampoo_beta"] = _track3_cd_timescale_beta(
                    group["shampoo_beta"], shampoo_beta_om_mult, batch_ratio
                )
            if "precond_lr" in group:
                group["precond_lr"] = group["precond_lr"] * precond_lr_mult

    if dist.get_rank() == 0:
        print0(
            "TRACK3_CORE_CD "
            + f"case={os.environ.get('TRACK3_CASE_ID', '')} "
            + f"coord={os.environ.get('TRACK3_COORD', 'center')} "
            + f"recipe={recipe} batch_size={batch_size} batch_ratio={batch_ratio:.6g} "
            + f"lr_scale={lr_scale:.6g} "
            + f"matrix_lr_mult={matrix_lr_mult:.6g} aux_lr_mult={aux_lr_mult:.6g} "
            + f"matrix_wd_peak={matrix_wd_peak:.6g} aux_wd_peak={aux_wd_peak:.6g} "
            + f"wd_warmup_frac={os.environ.get('TRACK3_WD_WARMUP_FRAC', '0')} "
            + f"matrix_beta1_om_mult={matrix_beta1_om_mult:.6g} "
            + f"matrix_beta2_om_mult={matrix_beta2_om_mult:.6g} "
            + f"precond_lr_mult={precond_lr_mult:.6g} "
            + f"shampoo_beta_om_mult={shampoo_beta_om_mult:.6g} "
            + f"aux_beta1_om_mult={aux_beta1_om_mult:.6g} "
            + f"aux_beta2_om_mult={aux_beta2_om_mult:.6g} "
            + f"aux_cooldown_frac={os.environ.get('TRACK3_AUX_COOLDOWN_FRAC', '0.4')}",
            console=True,
        )
'''


SEED_BLOCK = r'''
_track3_seed = int(os.environ.get("TRACK3_SEED", os.environ.get("KL_SOAP_SEED", "1")))
torch.manual_seed(_track3_seed)
torch.cuda.manual_seed_all(_track3_seed)
print0(f"TRACK3_CORE_CD seed={_track3_seed}", console=True)
'''


def _read_recipe_source(recipe: str) -> str:
    meta = RECIPES[recipe]
    text = (ROOT / meta["path"]).read_text()
    if meta.get("logfile"):
        marker = "\n" + "=" * 100 + "\n"
        if marker not in text:
            raise RuntimeError(f"Could not split logfile source for {recipe}")
        text = text.split(marker, 1)[0]
    return text


def _replace_block(text: str, start: str, end: str, replacement: str) -> str:
    start_index = text.index(start)
    end_index = text.index(end, start_index)
    return text[:start_index] + replacement.strip("\n") + text[end_index:]


def _patch_lionh_from_adamh(text: str) -> str:
    text = text.replace("train_gpt_simple_adamh.py", "train_gpt_simple_lionh.py", 1)
    text = text.replace("AdamH", "LionH")
    text = _replace_block(
        text,
        "class LionH(torch.optim.Optimizer):",
        "\n\n\n########################################\n#                Setup",
        LIONH_CLASS,
    )
    text = re.sub(
        r"optimizer2 = LionH\(\[p for p in model\.blocks\.parameters\(\) if p\.ndim >= 2\],\n"
        r"\s+lr=matrix_lr, betas=\(0\.9, 0\.95\), eps=1e-8\)",
        'optimizer2 = LionH([p for p in model.blocks.parameters() if p.ndim >= 2],\n'
        '                   lr=matrix_lr, betas=(0.9, 0.95), eps=1e-10)',
        text,
        count=1,
    )
    return text


def _patch_shampoo_betas(text: str) -> str:
    pattern = r"(?m)^(?P<indent>\s*)shampoo_beta2\s*=\s*(?P<value>[0-9.eE+-]+)\s*$"

    def repl(match: re.Match[str]) -> str:
        value = match.group("value")
        indent = match.group("indent")
        return (
            f'{indent}_track3_ratio = float(os.environ.get("TRACK3_BATCH_RATIO", "1.0"))\n'
            f'{indent}_track3_matrix_b2_mult = float(os.environ.get("TRACK3_MATRIX_BETA2_OM_MULT", "1.0"))\n'
            f'{indent}_track3_shampoo_b2_center = ({value}) ** _track3_ratio\n'
            f'{indent}shampoo_beta2 = float(os.environ.get("TRACK3_SHAMPOO_BETA2", '
            f'str(min(0.9999, max(0.0, 1.0 - (1.0 - _track3_shampoo_b2_center) * _track3_matrix_b2_mult)))))'
        )

    text, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one shampoo_beta2 assignment, found {count}")
    text = re.sub(
        r"betas=\(0\.9,\s*shampoo_beta2\)",
        'betas=(float(os.environ.get("TRACK3_SHAMPOO_BETA1", '
        'str(min(0.9999, max(0.0, 1.0 - (1.0 - 0.9) '
        '* float(os.environ.get("TRACK3_MATRIX_BETA1_OM_MULT", "1.0"))))))), shampoo_beta2)',
        text,
        count=1,
    )
    return text


def _patch_shampooh_from_rohan(text: str) -> str:
    text = text.replace("train_gpt_simple.py", "train_gpt_simple_shampooh.py", 1)
    text = text.replace("DistributedShampoo(\n        [p for p in model.blocks.parameters() if p.ndim >= 2],", "ShampooH(\n        [p for p in model.blocks.parameters() if p.ndim >= 2],", 1)
    text = text.replace("lr=1e-2,", "lr=0.018,", 1)
    text = text.replace("weight_decay=0.1,", "weight_decay=0.0,", 1)
    text = _patch_shampoo_betas(text)
    text = text.replace(
        "    # create the optimizer(s)\n",
        "    # H-family hidden matrices must have non-zero Frobenius radius.\n"
        "    # Rohan's original Shampoo recipe zeroes projection weights, which is fine\n"
        "    # for additive Shampoo but freezes those matrices under hyperball projection.\n"
        "    for name, p in model.named_parameters():\n"
        "        if name.endswith(\".attn.proj.weight\"):\n"
        "            p.data.normal_(std=0.33**0.5 / p.size(-1)**0.5).mul_(1.25)\n"
        "        elif name.endswith(\".mlp.proj.weight\"):\n"
        "            p.data.normal_(std=0.33**0.5 / p.size(-1)**0.5).mul_(3.0)\n"
        "        elif name.endswith(\".mlp.fc.weight\"):\n"
        "            p.data.mul_(1.5)\n"
        "\n"
        "    # create the optimizer(s)\n",
        1,
    )
    marker = "\n\ndef shampoo_distributed_config():"
    if marker not in text:
        raise RuntimeError("Could not find shampoo_distributed_config marker")
    text = text.replace(marker, "\n" + SHAMPOOH_CLASS + marker, 1)
    text = text.replace(
        "    optimizers = [optimizer1, optimizer2]\n",
        "    optimizers = [optimizer1, optimizer2]\n"
        "    for group in optimizer1.param_groups:\n"
        "        group[\"schedule_type\"] = \"aux\"\n"
        "        group[\"cooldown_frac\"] = 0.4\n"
        "    for group in optimizer2.param_groups:\n"
        "        group[\"schedule_type\"] = \"h\"\n"
        "        group[\"cooldown_frac\"] = 1.0\n",
        1,
    )
    schedule_pattern = (
        r"(?ms)^    # learning rate schedule: stable then decay\n"
        r"    def set_hparams\(step, cooldown_frac=0\.7\):\n"
        r".*?\n\n\n    ########################################"
    )
    schedule_replacement = """    # learning rate schedule: stable then decay
    def set_hparams(step):
        progress = step / train_steps
        assert 0 <= progress < 1
        for opt in optimizers:
            for group in opt.param_groups:
                cooldown_frac = group["cooldown_frac"]
                if progress < 1 - cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / cooldown_frac
                group["lr"] = group["initial_lr"] * eta


    ########################################"""
    text, count = re.subn(schedule_pattern, schedule_replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one Rohan Shampoo schedule block, found {count}")
    return text


def _patch_psgdh_from_pr316(text: str) -> str:
    """Adapt the merged PR #316 PSGD baseline to the Track-3 CD contract.

    The optimizer equations remain unchanged.  This transform only exposes the
    fixed-token horizon and the two optimizer schedules to the existing
    environment-driven materializer.
    """

    text = text.replace("train_gpt_simple.py", "train_psgdh_track3.py", 1)
    import_marker = "import os\n"
    if text.count(import_marker) != 1:
        raise RuntimeError("Expected one PSGD os import")
    text = text.replace(
        import_marker,
        import_marker
        # A separate cache per rank makes all eight ranks compile the same
        # kernels independently.  At campaign scale that can leave a subset
        # of workers in Inductor for many minutes.  PyTorch's cache is built
        # for concurrent readers/writers, so share it by default and retain
        # rank isolation only as an explicit diagnostic override.
        + 'if os.environ.get("TRACK3_ISOLATE_INDUCTOR_CACHE", "0") == "1":\n'
        + '    _track3_local_rank = os.environ.get("LOCAL_RANK", "0")\n'
        + '    os.environ.setdefault(\n'
        + '        "TORCHINDUCTOR_CACHE_DIR",\n'
        + '        f"/tmp/torchinductor_track3_rank_{_track3_local_rank}",\n'
        + '    )\n',
        1,
    )
    aliased_padding = (
        "            params_pad = params + [torch.empty_like(params[-1])] * "
        "(world_size - len(params) % world_size)\n"
    )
    if text.count(aliased_padding) != 1:
        raise RuntimeError("Expected one PSGD aliased padding expression")
    text = text.replace(
        aliased_padding,
        "            padding = (-len(params)) % world_size\n"
        "            params_pad = params + [\n"
        "                torch.empty_like(params[-1]) for _ in range(padding)\n"
        "            ]\n",
        1,
    )
    skipped_collective = (
        "                    if grad is None:\n"
        "                        continue\n"
    )
    if text.count(skipped_collective) != 1:
        raise RuntimeError("Expected one PSGD missing-gradient branch")
    text = text.replace(
        skipped_collective,
        "                    if grad is None:\n"
        "                        raise RuntimeError(\n"
        "                            \"PSGD matrix parameter is missing a gradient\"\n"
        "                        )\n",
        1,
    )
    optimizer_marker = "optimizers = [optimizer1, optimizer2]\n"
    if text.count(optimizer_marker) != 1:
        raise RuntimeError("Expected one PSGD optimizer list")
    text = text.replace(
        optimizer_marker,
        optimizer_marker
        + "for group in optimizer1.param_groups:\n"
        + "    group[\"schedule_type\"] = \"aux\"\n"
        + "    group[\"cooldown_frac\"] = float(os.environ.get(\"TRACK3_AUX_COOLDOWN_FRAC\", \"0.5\"))\n"
        + "for group in optimizer2.param_groups:\n"
        + "    group[\"schedule_type\"] = \"h\"\n"
        + "    group[\"cooldown_frac\"] = float(os.environ.get(\"TRACK3_H_COOLDOWN_FRAC\", \"1.0\"))\n"
        + "for opt in optimizers:\n"
        + "    for group in opt.param_groups:\n"
        + "        group[\"initial_lr\"] = group[\"lr\"]\n",
        1,
    )

    schedule_start = text.index("def get_psgd_lr(step: int):")
    schedule_end = text.index(
        "\n\n########################################\n#        Training and Validation",
        schedule_start,
    )
    schedule = '''def set_hparams(step: int):
    progress = step / train_steps
    assert 0 <= progress < 1
    for opt in optimizers:
        for group in opt.param_groups:
            cooldown_frac = group["cooldown_frac"]
            if progress < 1 - cooldown_frac:
                eta = 1.0
            else:
                eta = (1 - progress) / cooldown_frac
            group["lr"] = group["initial_lr"] * eta
'''
    text = text[:schedule_start] + schedule + text[schedule_end:]

    old_step_schedule = '''    for group in optimizer2.param_groups:
        group["lr"] = get_psgd_lr(step)
    adam_scale = get_adam_lr_scale(step)
    for group, base_lr in zip(optimizer1.param_groups, adam_base_lrs):
        group["lr"] = base_lr * adam_scale
'''
    if text.count(old_step_schedule) != 1:
        raise RuntimeError("Expected one PSGD per-step schedule block")
    return text.replace(old_step_schedule, "    set_hparams(step)\n", 1)


def _apply_recipe_transform(recipe: str, text: str) -> str:
    transform = RECIPES[recipe].get("transform")
    if transform == "lionh_from_adamh":
        return _patch_lionh_from_adamh(text)
    if transform == "shampooh_from_rohan":
        return _patch_shampooh_from_rohan(text)
    if transform == "psgdh_from_pr316":
        return _patch_psgdh_from_pr316(text)
    if transform:
        raise RuntimeError(f"Unknown transform {transform!r}")
    return text


def _replace_assignment(text: str, name: str, expr: str) -> str:
    pattern = rf"(?m)^(?P<indent>\s*){re.escape(name)}\s*=\s*.+$"
    replacement = rf"\g<indent>{name} = {expr}"
    new, count = re.subn(pattern, replacement, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one assignment for {name}, found {count}")
    return new


def _replace_train_steps(text: str) -> str:
    pattern = r"(?m)^(?P<indent>\s*)train_steps\s*=\s*(?P<value>\d+)\s*$"

    def repl(match: re.Match[str]) -> str:
        value = match.group("value")
        return (
            f'{match.group("indent")}train_steps = '
            f'int(os.environ.get("TRACK3_TRAIN_STEPS", "{value}"))'
        )

    new, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one train_steps assignment, found {count}")
    return new


def _insert_after_initial_lr(text: str) -> str:
    pattern = (
        r"(?m)^(?P<indent>\s*)for opt in optimizers:\n"
        r"(?P=indent)    for group in opt\.param_groups:\n"
        r"(?P=indent)        group\[\"initial_lr\"\] = group\[\"lr\"\]\n"
    )

    def repl(match: re.Match[str]) -> str:
        block = match.group(0)
        indent = match.group("indent")
        helper = "\n".join(
            (indent + line if line else "")
            for line in INJECT_HELPER.strip("\n").splitlines()
        )
        return block + "\n" + helper + f"\n{indent}_track3_cd_apply_scaled_hparams(optimizers)\n"

    new, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one initial_lr loop, found {count}")
    return new


def _patch_psgdh_serial_compile_warmup(text: str) -> str:
    """Compile PSGD kernels rank by rank before the measured training loop."""

    marker = "_track3_cd_apply_scaled_hparams(optimizers)\n"
    if text.count(marker) != 1:
        raise RuntimeError("Expected one PSGD scaled-hparam application")
    warmup = r'''

@torch.no_grad()
def _track3_serial_psgd_compile_warmup():
    """Compile PSGD kernels one rank at a time without touching training state."""
    if os.environ.get("TRACK3_SERIAL_PSGD_COMPILE_WARMUP", "1") != "1":
        return
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    params = optimizer2.param_groups[0]["params"]
    precond_lr = optimizer2.param_groups[0]["precond_lr"]
    for compile_rank in range(world_size):
        if rank == compile_rank:
            seen_shapes = set()
            for base_i in range(0, len(params), world_size):
                param_i = base_i + rank
                if param_i >= len(params):
                    continue
                shape = tuple(params[param_i].squeeze().shape)
                if shape in seen_shapes:
                    continue
                seen_shapes.add(shape)
                probe = torch.ones(shape, dtype=torch.float32, device=device)
                Q0, Q1 = init_psgd(probe)
                psgd_update_precond(
                    Q0, Q1, probe, torch.ones_like(probe),
                    precond_lr=precond_lr,
                )
                torch.cuda.synchronize()
                print0(
                    f"TRACK3_PSGD_COMPILE_WARMUP rank={rank} shape={shape}",
                    console=True,
                )
        dist.barrier()

_track3_serial_psgd_compile_warmup()
'''
    return text.replace(marker, marker + warmup, 1)


def _patch_decoupled_weight_decay(text: str) -> str:
    text = re.sub(
        r'(?m)^(?P<indent>\s*)scale_invariant_update_\(p, update, group\["lr"\]\)$',
        r'\g<indent>scale_invariant_update_(p, update, group["lr"])'
        "\n"
        r'\g<indent>_track3_cd_apply_decoupled_weight_decay(p, group)',
        text,
    )
    text = re.sub(
        r'(?m)^(?P<indent>\s*)group\["lr"\] = group\["initial_lr"\] \* eta$',
        r'\g<indent>group["lr"] = group["initial_lr"] * eta'
        "\n"
        r'\g<indent>_track3_cd_set_scheduled_weight_decay(group, step, train_steps)',
        text,
    )
    return text


def _patch_cooldown(text: str) -> str:
    text = text.replace(
        'cooldown_frac = 1.0 if group["schedule_type"] == "h" else 0.4',
        'cooldown_frac = float(os.environ.get("TRACK3_H_COOLDOWN_FRAC", "1.0")) '
        'if group["schedule_type"] == "h" else '
        'float(os.environ.get("TRACK3_AUX_COOLDOWN_FRAC", "0.4"))',
    )
    text = text.replace(
        '                cooldown_frac = 1.0\n',
        '                cooldown_frac = float(os.environ.get("TRACK3_H_COOLDOWN_FRAC", "1.0"))\n',
    )
    text = text.replace(
        '                cooldown_frac = 0.4\n',
        '                cooldown_frac = float(os.environ.get("TRACK3_AUX_COOLDOWN_FRAC", "0.4"))\n',
    )
    text = text.replace(
        'group["cooldown_frac"] = 1.0',
        'group["cooldown_frac"] = float(os.environ.get("TRACK3_H_COOLDOWN_FRAC", "1.0"))',
    )
    text = text.replace(
        'group["cooldown_frac"] = 0.4',
        'group["cooldown_frac"] = float(os.environ.get("TRACK3_AUX_COOLDOWN_FRAC", "0.4"))',
    )
    return text


def _patch_process_group_timeout(text: str) -> str:
    if "import datetime\n" not in text:
        text = text.replace("import time\n", "import time\nimport datetime\n", 1)
    pattern = r"dist\.init_process_group\((?P<args>[^)\n]*device_id=device[^)\n]*)\)"

    def repl(match: re.Match[str]) -> str:
        args = match.group("args")
        if "timeout=" in args:
            return match.group(0)
        return (
            '_track3_dist_timeout = datetime.timedelta(minutes=int('
            'os.environ.get("TRACK3_DIST_TIMEOUT_MINUTES", "180")))\n'
            f"dist.init_process_group({args}, timeout=_track3_dist_timeout)\n"
            "dist.distributed_c10d._set_pg_timeout(_track3_dist_timeout)\n"
            'if dist.get_rank() == 0:\n'
            '    print(f"TRACK3_DIST_TIMEOUT configured={_track3_dist_timeout}", flush=True)'
        )

    text, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise RuntimeError(f"Expected one dist.init_process_group call, found {count}")
    return text


def _patch_initial_parameter_sync(text: str) -> str:
    """Replace the per-parameter NCCL broadcast with an exact init audit.

    Every Track-3 core trainer resets the same CPU/CUDA seed before building
    the model, so all ranks already initialize byte-identical parameters.  A
    long sequence of one NCCL broadcast per parameter has nevertheless made a
    stalled A100 collective appear as one rank remaining in a broadcast while
    the other seven expose frames in the following barrier.  Removing the
    redundant tensor collectives both narrows that failure surface and gives
    an explicit, fail-closed initialization invariant.

    Hashing the exact parameter bytes on every rank and gathering only the
    small digests preserves the initialization contract without mutating a
    single tensor.  It also fails closed if a future source introduces
    rank-dependent initialization instead of silently training divergent
    models.  The training clock starts after this block, so the audit does not
    alter the learning-rate schedule or reported training time.
    """

    pattern = (
        r"(?m)^(?P<indent>\s*)for p in model\.parameters\(\):\n"
        r"(?P=indent)    dist\.broadcast\(p\.detach\(\), 0\)\n"
    )

    def render_audit(indent: str) -> str:
        lines = [
            "import hashlib as _track3_hashlib",
            "_track3_init_digest = _track3_hashlib.sha256()",
            "for _track3_name, _track3_param in model.named_parameters():",
            "    _track3_init_digest.update(_track3_name.encode(\"utf-8\"))",
            "    _track3_init_digest.update(str(_track3_param.dtype).encode(\"ascii\"))",
            "    _track3_init_digest.update(str(tuple(_track3_param.shape)).encode(\"ascii\"))",
            "    _track3_init_digest.update(",
            "        _track3_param.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()",
            "    )",
            "_track3_init_digests = [None] * dist.get_world_size()",
            "dist.all_gather_object(",
            "    _track3_init_digests, _track3_init_digest.hexdigest()",
            ")",
            "if len(set(_track3_init_digests)) != 1:",
            "    raise RuntimeError(",
            "        f\"Rank-dependent model initialization: {_track3_init_digests}\"",
            "    )",
            "print0(",
            "    f\"TRACK3_INIT_DIGEST {_track3_init_digests[0]}\", console=True",
            ")",
        ]
        return "\n".join(indent + line if line else "" for line in lines) + "\n"

    def repl(match: re.Match[str]) -> str:
        return render_audit(match.group("indent"))

    patched, count = re.subn(pattern, repl, text, count=1)
    if count == 0:
        # The standalone AdamW source zeroes the projection and broadcasts
        # every parameter inside one named-parameter loop.  Preserve the
        # projection-zeroing pass, remove only its redundant broadcasts, and
        # audit the final initialized model immediately afterward.
        named_pattern = (
            r'(?m)^(?P<indent>\s*)for name, p in model\.named_parameters\(\):\n'
            r'(?P=indent)    if "proj" in name:\n'
            r'(?P=indent)        p\.data\.zero_\(\)\n'
            r'(?P=indent)    dist\.broadcast\(p\.detach\(\), 0\)\n'
        )

        def named_repl(match: re.Match[str]) -> str:
            indent = match.group("indent")
            return (
                f"{indent}for name, p in model.named_parameters():\n"
                f'{indent}    if "proj" in name:\n'
                f"{indent}        p.data.zero_()\n"
                + render_audit(indent)
            )

        patched, count = re.subn(named_pattern, named_repl, text, count=1)
    if count != 1:
        raise RuntimeError(
            f"Expected one initial parameter broadcast loop, found {count}"
        )
    return patched


def _patch_strict_collective_completion(text: str) -> str:
    """Optionally fence gradient collectives and the complete optimizer step.

    NCCL calls are host-asynchronous even when ``async_op=False``.  On the
    A100 cohort, LionH has exhibited cross-rank queue skew: one rank reaches
    the next gradient reduction while the other ranks still expose Python
    frames in the previous optimizer step, after which the GPUs spin without
    reaching another validation boundary.

    The two completion sites are independently configurable.  A device fence
    after every gradient reduction and after the complete optimizer step
    preserves the arithmetic while preventing a rank from queueing a later
    collective before its peers.  Other recipes retain the legacy defaults
    through ``TRACK3_STRICT_COLLECTIVE_COMPLETION``.
    """

    reduce_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)dist\.all_reduce\(p\.grad, op=dist\.ReduceOp\."
        r"(?P<op>SUM|AVG)\)\n"
    )

    def replace_reduce(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f"{indent}dist.all_reduce(p.grad, op=dist.ReduceOp.{match.group('op')})\n"
            f'{indent}if os.environ.get("TRACK3_GRADIENT_COLLECTIVE_COMPLETION", os.environ.get("TRACK3_STRICT_COLLECTIVE_COMPLETION", "1")) == "1":\n'
            f"{indent}    torch.cuda.synchronize()\n"
        )

    text, reduce_count = reduce_pattern.subn(replace_reduce, text, count=1)
    gradient_phase_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)set_hparams\(step\)\n"
    )

    def replace_gradient_phase(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f'{indent}if os.environ.get("TRACK3_GRADIENT_PHASE_COMPLETION", "0") == "1":\n'
            f"{indent}    torch.cuda.synchronize()\n"
            f"{indent}set_hparams(step)\n"
        )

    text, gradient_phase_count = gradient_phase_pattern.subn(
        replace_gradient_phase, text, count=1
    )
    step_pattern = re.compile(
        r"(?m)^(?P<indent>\s*)model\.zero_grad\(set_to_none=True\)\n"
    )

    def replace_step(match: re.Match[str]) -> str:
        indent = match.group("indent")
        return (
            f'{indent}if os.environ.get("TRACK3_OPTIMIZER_STEP_COMPLETION", os.environ.get("TRACK3_STRICT_COLLECTIVE_COMPLETION", "1")) == "1":\n'
            f"{indent}    torch.cuda.synchronize()\n"
            f"{indent}model.zero_grad(set_to_none=True)\n"
        )

    text, step_count = step_pattern.subn(replace_step, text, count=1)
    if reduce_count != 1 or gradient_phase_count != 1 or step_count != 1:
        raise RuntimeError(
            "Expected one strict collective completion insertion site; "
            f"gradient_reductions={reduce_count}, "
            f"gradient_phases={gradient_phase_count}, optimizer_steps={step_count}"
        )
    return text


def _patch_shampoo_sync_fences(text: str) -> str:
    """Optionally align ranks around cold Shampoo compute without changing math."""
    before_reduce = (
        "        for i in range(len(inputs) // mbs):\n"
        "            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()\n"
        "        for name, p in model.named_parameters():\n"
    )
    after_reduce = (
        "        for i in range(len(inputs) // mbs):\n"
        "            model(inputs[i*mbs:(i+1)*mbs], targets[i*mbs:(i+1)*mbs]).backward()\n"
        "        if os.environ.get(\"TRACK3_SHAMPOO_SYNC_FENCES\", \"0\") == \"1\":\n"
        "            dist.barrier()\n"
        "        for name, p in model.named_parameters():\n"
    )
    before_step_end = (
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        model.zero_grad(set_to_none=True)\n"
    )
    after_step_end = (
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        if os.environ.get(\"TRACK3_SHAMPOO_SYNC_FENCES\", \"0\") == \"1\":\n"
        "            dist.barrier()\n"
        "        model.zero_grad(set_to_none=True)\n"
    )
    if text.count(before_reduce) != 1 or text.count(before_step_end) != 1:
        raise RuntimeError("Expected one Shampoo training-loop fence insertion site")
    return text.replace(before_reduce, after_reduce, 1).replace(
        before_step_end, after_step_end, 1
    )


def materialize(recipe: str) -> str:
    text = _read_recipe_source(recipe)
    if RECIPES[recipe].get("exact_passthrough"):
        # Baseline parity is deliberately stricter than the CD contract.  The
        # PR #316 source must run byte-for-byte unchanged so that schedule,
        # initialization, collectives, and optimizer equations are all part of
        # the reproduction check.  The surrounding launcher may still restore
        # data and persist logs, neither of which changes the training program.
        return text
    if RECIPES[recipe].get("uses_shampoo"):
        text = text.replace(
            "import sys\n",
            f'import sys\nsys.path.insert(0, os.path.join(os.getcwd(), "{SHAMPOO_VENDOR}"))\n',
            1,
        )
    text = _apply_recipe_transform(recipe, text)
    text = _replace_assignment(
        text,
        "batch_size",
        'int(os.environ.get("TRACK3_BATCH_SIZE", str(8 * 64 * 1024)))',
    )
    text = _replace_assignment(
        text,
        "mbs",
        'max(1, min(64, batch_size // (dist.get_world_size() * 1024)))',
    )
    text = _replace_train_steps(text)
    text = _patch_cooldown(text)
    text = _patch_decoupled_weight_decay(text)
    text = _patch_process_group_timeout(text)
    text = _patch_initial_parameter_sync(text)
    if RECIPES[recipe].get("uses_shampoo"):
        text = _patch_shampoo_sync_fences(text)
    text = _patch_strict_collective_completion(text)
    text = text.replace("val_tokens = 20 * 524288\n", SEED_BLOCK + "\nval_tokens = 20 * 524288\n", 1)
    text = _insert_after_initial_lr(text)
    if recipe == "psgdh_core":
        text = _patch_psgdh_serial_compile_warmup(text)
    header = (
        "# Materialized by records/track_3_optimization/batch_size_cd/materialize_core_hparam.py\n"
        f"# recipe={recipe}\n"
    )
    return header + text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", default=os.environ.get("TRACK3_RECIPE", "muonh_core"), choices=sorted(RECIPES))
    parser.add_argument("--write-script", type=Path, required=True)
    args = parser.parse_args()

    code = materialize(args.recipe)
    args.write_script.parent.mkdir(parents=True, exist_ok=True)
    args.write_script.write_text(code)
    print(args.write_script)


if __name__ == "__main__":
    main()
