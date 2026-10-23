"""Split integrity: disjoint, complete, reproducible; saved split files have no leakage."""

from pathlib import Path

import pytest

from grasp_planner.data.splits import block_split, check_disjoint, image_wise_split, load_split, make_split

IDS = [f"{i:04d}" for i in range(100, 985)]


def test_image_split_disjoint_complete_reproducible():
    a = image_wise_split(IDS, (0.7, 0.15, 0.15), 42)
    b = image_wise_split(IDS, (0.7, 0.15, 0.15), 42)
    assert a == b
    check_disjoint(a)
    assert sorted(a["train"] + a["val"] + a["test"]) == IDS
    assert image_wise_split(IDS, (0.7, 0.15, 0.15), 1) != a


def test_block_split_gap_and_contiguity():
    s = block_split(IDS, (0.7, 0.15, 0.15), 42, block_size=20, gap=2)
    check_disjoint(s)
    assert sorted(s["train"] + s["val"] + s["test"] + s["dropped_gap"], key=int) == IDS
    owner = {i: k for k, v in s.items() for i in v}
    # every pair of neighbouring kept scenes belonging to different splits is separated by >= 2*gap drops
    kept = [i for i in IDS if owner[i] != "dropped_gap"]
    for a, b in zip(kept, kept[1:], strict=False):
        if owner[a] != owner[b]:
            assert int(b) - int(a) > 2 * 2


def test_check_disjoint_raises():
    with pytest.raises(AssertionError):
        check_disjoint({"train": ["1", "2"], "test": ["2"]})


def test_saved_split_files_are_leakage_free():
    split_dir = Path(__file__).resolve().parents[1] / "experiments" / "splits"
    files = sorted(split_dir.glob("*.json"))
    if not files:
        pytest.skip("split files not generated yet")
    for f in files:
        s = load_split(f)  # raises on overlap
        assert s["counts"]["test"] > 0 and s["counts"]["val"] > 0


def test_make_split_metadata():
    s = make_split(IDS, "image", (0.7, 0.15, 0.15), 3)
    assert s["seed"] == 3 and s["counts"]["train"] == len(s["splits"]["train"])
