"""Annotation parsing and dataset discovery (synthetic files + real dataset if configured)."""

import numpy as np
import pytest

from grasp_planner.data.cornell import discover_samples, parse_rectangle_file
from grasp_planner.data.grasp import rect_to_grasp


def test_parse_valid_and_reports_bad(tmp_path):
    f = tmp_path / "pcd0001cpos.txt"
    f.write_text(
        "253 319.7 \n309 324 \n307 350 \n251 345.7\n"  # real first rectangle of pcd0100cpos.txt
        "NaN NaN\nNaN NaN\n1 1\n2 2\n"  # NaN rectangle -> rejected
        "600 10\n700 10\n700 20\n600 20\n"  # outside 640-wide image -> rejected
    )
    r = parse_rectangle_file(f)
    assert r.file_error is None
    assert r.rects.shape == (1, 4, 2)
    assert r.rects[0, 0] == pytest.approx([253, 319.7])
    assert [i for i, _ in r.rejected] == [1, 2]
    g = rect_to_grasp(r.rects[0])
    assert g.width == pytest.approx(np.hypot(56, 4.3))


def test_parse_wrong_line_count(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("1 2\n3 4\n5 6\n")
    r = parse_rectangle_file(f)
    assert r.file_error and "multiple of 4" in r.file_error
    assert r.rects.shape == (0, 4, 2)


def test_discover_synthetic(synthetic_root):
    samples, excl = discover_samples(synthetic_root)
    assert len(samples) == 12 and excl == []
    assert all(s.depth_path is not None for s in samples)


def test_discover_reports_missing_annotation(synthetic_root):
    (synthetic_root / "01" / "pcd0100cpos.txt").rename(synthetic_root / "01" / "moved.txt")
    samples, excl = discover_samples(synthetic_root)
    assert len(samples) == 11
    assert excl == [{"sample_id": "0100", "folder": "01", "reason": "missing cpos.txt"}]


@pytest.mark.dataset
def test_real_cornell_discovery(cornell_root):
    samples, excl = discover_samples(cornell_root)
    assert len(samples) + len(excl) == 885
    s = next(s for s in samples if s.sample_id == "0100")
    assert len(s.pos) == 4  # pcd0100cpos.txt has 16 lines
    assert s.pos[0, 0] == pytest.approx([253, 319.7])
    assert np.isfinite(np.concatenate([x.pos.ravel() for x in samples])).all()
