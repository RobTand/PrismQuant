from pathlib import Path


DRIVER = (
    Path(__file__).resolve().parents[1]
    / "research/trellis_e2m1_highrate_2026-08-30/e2m1_highrate.py"
)


def test_active_checkout_precedes_locked_hull_snapshot_on_sys_path():
    source = DRIVER.read_text()
    locked_insert = "sys.path.insert(0, str(LOCKED_HULL_ROOT))"
    repo_insert = "sys.path.insert(0, str(REPO_ROOT))"

    assert source.index(locked_insert) < source.index(repo_insert)
