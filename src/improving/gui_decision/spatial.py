"""Spatial readout helpers for the single-screenshot Qwen decision adapter.

These helpers operate on merged *language-model image tokens*, not on the
unmerged ViT patch sequence. They do not change executable candidate boxes.
"""
from __future__ import annotations

import math

import torch


def last_valid_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return the last unmasked position for either left or right padding."""
    if attention_mask.ndim != 2 or attention_mask.shape[1] == 0:
        raise ValueError("attention_mask must have shape [B, L] with L > 0")
    valid = attention_mask.bool()
    if not bool(valid.any(dim=-1).all()):
        raise ValueError("Every sample must contain at least one unmasked token")
    positions = torch.arange(valid.shape[1], device=valid.device)
    return positions.expand_as(valid).masked_fill(~valid, -1).amax(dim=-1)


def restore_merged_grid(
    image_tokens: torch.Tensor,
    image_grid_thw_row: torch.Tensor,
    merge_size: int = 2,
) -> torch.Tensor:
    """Restore [H, W, D] from Qwen's row-major merged image-token sequence.

    ``image_grid_thw`` describes the raw patch grid. The processor groups each
    spatial merge block contiguously before the vision merger; the resulting
    language-model tokens are row-major over (H / merge, W / merge).
    """
    if image_tokens.ndim != 2:
        raise ValueError("image_tokens must have shape [N, D]")
    if not isinstance(merge_size, int) or isinstance(merge_size, bool) or merge_size < 1:
        raise ValueError("merge_size must be a positive integer")
    grid = torch.as_tensor(image_grid_thw_row).detach().cpu().reshape(-1)
    if grid.numel() != 3 or not bool(torch.isfinite(grid).all()):
        raise ValueError("image_grid_thw_row must contain finite [T, H, W]")
    values = grid.tolist()
    if any(float(value) != int(value) for value in values):
        raise ValueError("image_grid_thw values must be integers")
    temporal, raw_h, raw_w = (int(value) for value in values)
    if temporal != 1:
        raise ValueError("Version 1 accepts one static screenshot per sample (T == 1)")
    if raw_h <= 0 or raw_w <= 0 or raw_h % merge_size or raw_w % merge_size:
        raise ValueError("Raw image H and W must be positive multiples of merge_size")
    height, width = raw_h // merge_size, raw_w // merge_size
    if image_tokens.shape[0] != height * width:
        raise ValueError(
            f"Image-token count {image_tokens.shape[0]} does not match merged grid "
            f"{height} x {width}"
        )
    return image_tokens.reshape(height, width, image_tokens.shape[-1])


def pool_candidate_regions(
    grid: torch.Tensor,
    bounds: torch.Tensor,
    has_region: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool candidate boxes; global actions use the full-grid mean.

    Partly off-screen a11y boxes are legal. Only the pooling interval is clipped;
    the original action coordinates and geometry features remain untouched.
    Padded candidates receive zero vectors. As in GUI-JEPA, the upper grid index
    includes the cell containing the box's upper edge, and every ROI has at least
    one cell. This is a feature-pooling convention, not a click-label change.
    """
    if grid.ndim != 3 or min(grid.shape) < 1:
        raise ValueError("grid must have nonempty shape [H, W, D]")
    if bounds.ndim != 2 or bounds.shape[-1] != 4:
        raise ValueError("bounds must have shape [N, 4]")
    count = bounds.shape[0]
    if has_region.shape != (count,) or candidate_mask.shape != (count,):
        raise ValueError("Region and candidate masks must have shape [N]")
    height, width, dim = grid.shape
    global_feature = grid.mean(dim=(0, 1))
    output = []
    boxes = bounds.detach().cpu().tolist()
    regions = has_region.detach().cpu().bool().tolist()
    valid = candidate_mask.detach().cpu().bool().tolist()
    for index in range(count):
        if not valid[index]:
            output.append(grid.new_zeros(dim))
            continue
        if not regions[index]:
            output.append(global_feature)
            continue
        left, top, right, bottom = (float(value) for value in boxes[index])
        if not all(math.isfinite(value) for value in (left, top, right, bottom)):
            raise ValueError(f"Candidate {index} has a nonfinite box")
        if right <= left or bottom <= top:
            raise ValueError(f"Candidate {index} has a degenerate box")
        left, right = max(0.0, min(1.0, left)), max(0.0, min(1.0, right))
        top, bottom = max(0.0, min(1.0, top)), max(0.0, min(1.0, bottom))
        col0 = max(0, min(width - 1, int(left * width)))
        row0 = max(0, min(height - 1, int(top * height)))
        col1 = max(col0 + 1, min(width, int(right * width) + 1))
        row1 = max(row0 + 1, min(height, int(bottom * height) + 1))
        output.append(grid[row0:row1, col0:col1].mean(dim=(0, 1)))
    if not output:
        return grid.new_zeros((0, dim))
    return torch.stack(output, dim=0)
