"""Unit tests for morphohand.studies.runlib (the shared study-script plumbing)."""
from __future__ import annotations

import json
import os
import time

from morphohand.studies import runlib


def _mk_run(tmp_path, name, iters):
    run = tmp_path / name
    (run / "tensorboard").mkdir(parents=True)
    for i in iters:
        (run / "tensorboard" / f"model_{i}.pt").write_text("x")
    return run


def test_final_ckpt_orders_numerically(tmp_path):
    run = _mk_run(tmp_path, "run", [9, 10, 100, 2])
    assert runlib.final_ckpt(run).name == "model_100.pt"


def test_final_ckpt_tolerates_missing(tmp_path):
    assert runlib.final_ckpt(None) is None
    empty = tmp_path / "empty"
    (empty / "tensorboard").mkdir(parents=True)
    assert runlib.final_ckpt(empty) is None


def test_latest_run_by_mtime(tmp_path):
    old = tmp_path / "a_tag"
    new = tmp_path / "b_tag"
    old.mkdir(), new.mkdir()
    past = time.time() - 100
    os.utime(old, (past, past))
    assert runlib.latest_run("*tag", root=tmp_path) == new
    assert runlib.latest_run("*nomatch", root=tmp_path) is None


def test_iter_objheight_parses_trainer_log(tmp_path):
    log = tmp_path / "t.log"
    log.write_text(
        "Learning iteration 1/100\n  lift_height/object_height: 0.010\n"
        "Learning iteration 2/100\n  other: 1\n"
        "Learning iteration 3/100\n  lift_height/object_height: 0.085\n")
    assert runlib.iter_objheight(log) == {1: 0.010, 3: 0.085}
    assert runlib.iter_objheight(tmp_path / "missing.log") == {}


def test_best_a_ckpt_salvages_pre_collapse(tmp_path):
    run = _mk_run(tmp_path, "run", [10, 20, 30])
    log = tmp_path / "t.log"
    log.write_text(
        "Learning iteration 10/30\n  lift_height/object_height: 0.05\n"
        "Learning iteration 20/30\n  lift_height/object_height: 0.12\n"
        "Learning iteration 30/30\n  lift_height/object_height: 0.01\n")  # collapsed at the end
    ck, oh = runlib.best_a_ckpt(run, log)
    assert ck.name == "model_20.pt" and oh == 0.12


def test_best_a_ckpt_falls_back_to_final_without_log(tmp_path):
    run = _mk_run(tmp_path, "run", [10, 20])
    ck, oh = runlib.best_a_ckpt(run, tmp_path / "missing.log")
    assert ck.name == "model_20.pt" and oh is None


def test_record_store_resumes(tmp_path):
    path = tmp_path / "state.json"
    store = runlib.RecordStore(path, key_field="id")
    store.put({"id": "a", "x": 1})
    store.put({"id": "b", "x": 2})
    store.put({"id": "a", "x": 3})  # overwrite persists immediately
    reloaded = runlib.RecordStore(path, key_field="id")
    assert "a" in reloaded and reloaded.get("a")["x"] == 3
    assert len(reloaded) == 2
    assert json.loads(path.read_text()) == reloaded.values()


def test_json_state_roundtrip(tmp_path):
    path = tmp_path / "s.json"
    st = runlib.JsonState(path)
    assert st.get("a_ckpt") is None
    st.update(a_ckpt="x", **{"b_ckpt_k0": "y"})
    st.save()
    assert runlib.JsonState(path).get("b_ckpt_k0") == "y"


def test_txt_report_header_once(tmp_path):
    path = tmp_path / "r.txt"
    runlib.TxtReport(path, "# header\n").line("row1", echo=False)
    runlib.TxtReport(path, "# DIFFERENT header\n").line("row2", echo=False)
    assert path.read_text() == "# header\nrow1\nrow2\n"


def test_sentinel(tmp_path):
    s = runlib.Sentinel(tmp_path / "X.DONE")
    s.clear()  # no-op when absent
    s.write("8 designs")
    assert "8 designs" in s.path.read_text()
    s.clear()
    assert not s.path.exists()


def test_warp_cache_env_is_fresh_per_call():
    e1, e2 = runlib.warp_cache_env({}), runlib.warp_cache_env({})
    assert e1["WARP_CACHE_PATH"] != e2["WARP_CACHE_PATH"]
    assert e1["MUJOCO_GL"] == "egl"
    assert os.path.isdir(e1["WARP_CACHE_PATH"])


def test_root_is_repo_root():
    assert (runlib.ROOT / "pyproject.toml").exists()
    assert (runlib.ROOT / "scripts").is_dir()
