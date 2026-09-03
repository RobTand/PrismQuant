"""Activation row sampling helpers.

Calibration captures should not keep the first rows seen from a forward
stream. That biases every downstream solver toward early tokens. The helper
below assigns each observed row an independent random priority and keeps the
top-k priorities, which is equivalent to uniform sampling without replacement
over all rows seen so far while staying bounded to ``max_rows`` storage.
"""
from __future__ import annotations

import torch


def update_priority_reservoir(
    current_rows: torch.Tensor | None,
    current_priorities: torch.Tensor | None,
    new_rows: torch.Tensor,
    *,
    max_rows: int,
    generator: torch.Generator | None = None,
    new_priorities: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Return an updated uniform row sample.

    ``current_priorities`` must contain the random priorities for
    ``current_rows`` and is kept on CPU. ``new_rows`` may live on CPU or CUDA;
    returned rows stay on the same device/dtype as the candidate row tensor.

    ``new_priorities`` lets the caller draw the incoming rows' priorities
    itself. A caller that hooks many tensors off one generator must draw for
    *every* hooked tensor, or a run that stores a subset consumes a different
    slice of the stream and the surviving rows stop being the rows the full
    run would have kept — see ``_LinearActivationCollector``. Exactly one of
    ``generator`` / ``new_priorities`` is required.
    """
    limit = int(max_rows)
    if limit <= 0 or new_rows.numel() == 0:
        return current_rows, current_priorities
    if new_rows.dim() != 2:
        raise ValueError("priority reservoir expects 2D row tensors")

    incoming = new_rows.detach()
    if (generator is None) == (new_priorities is None):
        raise ValueError(
            "priority reservoir requires exactly one of generator / "
            "new_priorities"
        )
    if new_priorities is None:
        new_priorities = torch.rand(
            int(incoming.shape[0]),
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
    else:
        new_priorities = new_priorities.detach().to(
            device="cpu", dtype=torch.float32
        )
        if int(new_priorities.shape[0]) != int(incoming.shape[0]):
            raise ValueError(
                "new_priorities length must match the incoming row count"
            )
    if current_rows is None:
        candidates = incoming
        priorities = new_priorities
    else:
        if current_priorities is None:
            raise ValueError("current_priorities required with current_rows")
        if current_rows.dim() != 2:
            raise ValueError("current_rows must be 2D")
        if int(current_rows.shape[1]) != int(incoming.shape[1]):
            raise ValueError(
                "current_rows and new_rows must have the same feature width"
            )
        candidates = torch.cat([current_rows, incoming], dim=0)
        priorities = torch.cat([current_priorities.cpu(), new_priorities], dim=0)

    if int(candidates.shape[0]) <= limit:
        return candidates.clone(), priorities.clone()

    _, keep_cpu = torch.topk(priorities, k=limit, largest=True, sorted=False)
    keep_device = keep_cpu.to(device=candidates.device)
    return (
        candidates.index_select(0, keep_device).clone(),
        priorities.index_select(0, keep_cpu).clone(),
    )
