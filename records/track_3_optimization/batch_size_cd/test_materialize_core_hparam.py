import re

from records.track_3_optimization.batch_size_cd import materialize_core_hparam as core


def transformed_recipes():
    return [
        recipe
        for recipe, metadata in core.RECIPES.items()
        if not metadata.get("exact_passthrough")
    ]


def test_core_recipes_audit_exact_seeded_initialization_without_broadcast_loop():
    for recipe in transformed_recipes():
        code = core.materialize(recipe)
        assert "for p in model.parameters():\n    dist.broadcast(p.detach(), 0)" not in code
        assert code.count("TRACK3_INIT_DIGEST") == 1
        assert code.count("dist.all_gather_object(") == 1
        assert code.count("Rank-dependent model initialization") == 1


def test_initialization_audit_precedes_training_clock_barrier():
    code = core.materialize("lionh_core")
    digest = code.index("TRACK3_INIT_DIGEST")
    training_clock = code.index("# start the clock")
    assert digest < training_clock
    assert "dist.barrier()" in code[digest:training_clock + 200]


def test_core_recipes_fence_gradient_collectives_and_optimizer_steps():
    for recipe in transformed_recipes():
        code = core.materialize(recipe)
        assert code.count("TRACK3_STRICT_COLLECTIVE_COMPLETION") == 2
        assert "_track3_gradient_works = []" in code
        assert 'TRACK3_EXPLICIT_GRADIENT_WORKS", "0"' in code
        assert "_track3_gradient_works.append(" in code
        assert "async_op=True" in code
        assert "for _track3_work in _track3_gradient_works:" in code
        assert "_track3_work.wait()" in code
        assert 'TRACK3_GRADIENT_PHASE_COMPLETION", "0"' in code
        assert re.search(
            r'(?m)^(?P<i>\s*)if os\.environ\.get\("TRACK3_OPTIMIZER_STEP_COMPLETION", '
            r'os\.environ\.get\("TRACK3_STRICT_COLLECTIVE_COMPLETION", "1"\)\) == "1":\n'
            r"(?P=i)    torch\.cuda\.synchronize\(\)\n"
            r"(?P=i)model\.zero_grad\(set_to_none=True\)",
            code,
        )


def test_psgdh_preserves_pr316_update_and_exposes_track3_controls():
    code = core.materialize("psgdh_core")
    assert "class PSGDKron(torch.optim.Optimizer):" in code
    assert "psgd_update_precond(" in code
    assert "step_update = (Q0.mT @ Q0) @ update @ (Q1.mT @ Q1)" in code
    assert 'TRACK3_PRECOND_LR_MULT' in code
    assert 'TRACK3_H_COOLDOWN_FRAC' in code
    assert 'TRACK3_AUX_COOLDOWN_FRAC' in code
    assert 'TRACK3_PSGD_EXPLICIT_GATHER_COMPLETION' in code
    assert "gather_work.wait()" in code
    gradient_wait_start = code.rindex(
        'if os.environ.get("TRACK3_EXPLICIT_GRADIENT_WORKS"'
    )
    gradient_wait = code[
        gradient_wait_start:
        code.index(
            'elif os.environ.get("TRACK3_GRADIENT_PHASE_COMPLETION"',
            gradient_wait_start,
        )
    ]
    assert "torch.cuda.synchronize()" not in gradient_wait
    gather_wait = code[
        code.index('if os.environ.get("TRACK3_PSGD_EXPLICIT_GATHER_COMPLETION"'):
        code.index("                else:\n", code.index("gather_work.wait()"))
    ]
    assert "torch.cuda.synchronize()" not in gather_wait
    assert "get_psgd_lr" not in code
    assert "get_adam_lr_scale" not in code
    assert code.count("_track3_cd_set_scheduled_weight_decay(group, step, train_steps)") == 2
    compile(code, "train_psgdh_track3.py", "exec")
