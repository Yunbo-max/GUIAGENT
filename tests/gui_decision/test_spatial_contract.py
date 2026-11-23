"""Authored spatial contracts; not a Qwen or GPU qualification run.

The fixtures encode independent screen regions with known values.  They detect
row/column swaps, pre-merge token use, and padding mistakes without downloading
or constructing a scientific model.
"""

import pytest
import torch

from improving.gui_decision.spatial import (
    last_valid_token_indices,
    pool_candidate_regions,
    restore_merged_grid,
)


def test_last_token_is_the_last_unmasked_position_for_both_padding_sides():
    attention_mask = torch.tensor(
        [[0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 1, 0]], dtype=torch.long
    )

    assert last_valid_token_indices(attention_mask).tolist() == [3, 1, 2]


def test_fully_padded_sequence_cannot_supply_a_query_feature():
    with pytest.raises(ValueError):
        last_valid_token_indices(torch.zeros((1, 4), dtype=torch.long))


def test_merged_tokens_restore_a_rectangular_screen_without_transposition():
    # A 4-by-6 pre-merge grid becomes two rows of three 2-by-2 regions.
    # These are already merged features, not the 24 raw vision tokens.
    tokens = torch.tensor([[10.0], [20.0], [30.0], [40.0], [50.0], [60.0]])

    grid = restore_merged_grid(tokens, torch.tensor([1, 4, 6]), merge_size=2)

    assert tuple(grid.shape) == (2, 3, 1)
    assert grid[0, 2, 0].item() == 30.0
    assert grid[1, 0, 0].item() == 40.0
    assert grid[1, 2, 0].item() == 60.0


@pytest.mark.parametrize(
    ("tokens", "grid_thw"),
    [
        (torch.zeros(24, 1), [1, 4, 6]),  # unmerged instead of merged tokens
        (torch.zeros(6, 1), [2, 4, 6]),  # video is outside this image-only contract
        (torch.zeros(6, 1), [1, 3, 8]),  # cannot form complete 2-by-2 regions
    ],
)
def test_wrong_vision_grid_contract_is_rejected(tokens, grid_thw):
    with pytest.raises(ValueError):
        restore_merged_grid(tokens, torch.tensor(grid_thw), merge_size=2)


def test_region_pooling_preserves_screen_axes_and_right_bottom_edge():
    # Two rows, four columns. The first two boxes lie strictly within one cell,
    # so this does not impose a new upper-edge pooling convention on the adapter.
    grid = torch.tensor(
        [[[10.0], [20.0], [30.0], [40.0]],
         [[50.0], [60.0], [70.0], [80.0]]]
    )
    bounds = torch.tensor(
        [[0.01, 0.51, 0.24, 0.99],  # bottom-left cell
         [0.51, 0.01, 0.74, 0.49],  # third cell of top row
         [0.99, 0.99, 1.0, 1.0],  # tiny target at inclusive screen edge
         [0.0, 0.0, 1.0, 1.0]]   # complete screen
    )

    pooled = pool_candidate_regions(
        grid,
        bounds,
        torch.ones(4, dtype=torch.bool),
        torch.ones(4, dtype=torch.bool),
    )

    torch.testing.assert_close(pooled[:, 0], torch.tensor([50.0, 30.0, 80.0, 45.0]))


def test_padding_region_does_not_leak_visual_content():
    grid = torch.tensor([[[3.0], [9.0]], [[15.0], [21.0]]])
    pooled = pool_candidate_regions(
        grid,
        torch.tensor([[0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 1.0]]),
        torch.tensor([True, True]),
        torch.tensor([True, False]),
    )

    assert pooled[0, 0].item() == 12.0
    assert pooled[1, 0].item() == 0.0


def test_global_action_uses_screen_mean_without_an_element_box():
    grid = torch.tensor([[[3.0], [9.0]], [[15.0], [21.0]]])

    pooled = pool_candidate_regions(
        grid,
        torch.zeros(1, 4),
        torch.tensor([False]),
        torch.tensor([True]),
    )

    assert pooled[0, 0].item() == 12.0


def test_partly_offscreen_pooling_does_not_rewrite_executable_bounds():
    grid = torch.tensor([[[10.0], [20.0]], [[30.0], [40.0]]])
    bounds = torch.tensor([[-0.25, 0.51, 0.24, 1.25]])
    original_bounds = bounds.clone()

    pooled = pool_candidate_regions(
        grid, bounds, torch.tensor([True]), torch.tensor([True])
    )

    assert pooled[0, 0].item() == 30.0
    torch.testing.assert_close(bounds, original_bounds)
