def calculate_multistep_lrs(base_lrs, milestones, gamma, completed_epochs):
    """Calculate the LR that should be active after completed_epochs."""
    decay_count = sum(
        int(milestone) <= int(completed_epochs) for milestone in milestones
    )
    factor = float(gamma) ** decay_count
    return [float(base_lr) * factor for base_lr in base_lrs]


def rebuild_multistep_scheduler(
    optimizer,
    base_lrs,
    milestones,
    gamma,
    completed_epochs,
):
    """Rebuild a resumed schedule from the current config.

    Optimizer checkpoints contain both Adam moments and the learning rate from
    the old run. Restore the moments first, then call this helper so the current
    YAML's base learning rates, milestones, and decay factor take precedence.
    """
    from torch.optim.lr_scheduler import MultiStepLR

    base_lrs = [float(value) for value in base_lrs]
    if len(optimizer.param_groups) != len(base_lrs):
        raise ValueError(
            "Optimizer parameter-group count does not match the current "
            f"configuration: {len(optimizer.param_groups)} vs. {len(base_lrs)}"
        )

    completed_epochs = int(completed_epochs)
    if completed_epochs < 0:
        raise ValueError(
            f"completed_epochs must be non-negative, got {completed_epochs}"
        )

    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["initial_lr"] = base_lr
        group["lr"] = base_lr

    scheduler = MultiStepLR(
        optimizer,
        milestones=list(milestones),
        gamma=float(gamma),
    )
    scheduler.step(completed_epochs)
    return scheduler
