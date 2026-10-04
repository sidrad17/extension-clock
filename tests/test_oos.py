"""Tests for the Gate 2 path (CLAUDE.md section 17): run_all.py --oos, src/oos_eval.py, src/data/oos_data.py.

* Before the gate2-frozen tag, `--oos` with no committed data/oos/ exits cleanly in every mode (GQH_DEV on or off)
  and makes no network request; every test-window download refuses without the tag or on a dirty tree.
* The run-once rule, the merge into results.json, the labels and the data view's splice.
* The pseudo-window check (`python run_all.py --oos-pseudo`): evaluate_window through a data view that splits the
  committed snapshot at 2022-09-30 equals the engines run directly on 2022-10-01..2024-09-30 (src/oos_check.py).
"""
import json
import socket

import pandas as pd
import pytest
import requests

import run_all
import src.trial_log as tl
from src import oos_eval
from src.data import databento_futures as dbf
from src.data import oos_data


@pytest.fixture
def network_log(monkeypatch):
    """Record every attempted network request (socket or requests) and refuse it."""
    calls = []

    def refuse(*args, **kwargs):
        calls.append(args[1:2] or kwargs)
        raise RuntimeError("network access in tests is not allowed")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(requests.Session, "request", refuse)
    return calls


@pytest.fixture
def before_tag(monkeypatch, tmp_path):
    """HEAD without gate2-frozen, no committed data/oos/, nothing staged, an empty run log (all in tmp_path)."""
    monkeypatch.setattr(tl, "_head_tags", lambda: [])
    monkeypatch.setattr(tl, "_dirty", lambda: False)
    monkeypatch.setattr(oos_data, "OOS_DATA_DIR", tmp_path / "oos")
    monkeypatch.setattr(oos_data, "STAGE_DIR", tmp_path / "stage")
    monkeypatch.setattr(oos_data, "VIEW_DIR", tmp_path / "view")
    monkeypatch.setattr(run_all, "OOS_LOG", tmp_path / "oos_run.log")
    monkeypatch.setattr(run_all, "OOS_MARK", tmp_path / "run_started.log")
    monkeypatch.setattr(run_all, "OOS_JSON", tmp_path / "results_oos.json")
    return tmp_path


@pytest.mark.parametrize("gqh_dev", ["1", ""])
def test_oos_before_tag_exits_cleanly_without_network(gqh_dev, monkeypatch, before_tag, network_log):
    monkeypatch.setenv("GQH_DEV", gqh_dev)
    with pytest.raises(SystemExit) as e:
        run_all.main(["--oos"])
    assert "gate2-frozen" in str(e.value.code) and "Nothing downloaded" in str(e.value.code)
    assert network_log == []
    assert not any(before_tag.iterdir())          # no log line, no staging, no view, no results


@pytest.mark.parametrize("gqh_dev", ["1", ""])
def test_oos_databento_ok_before_tag_also_refuses(gqh_dev, monkeypatch, before_tag, network_log):
    monkeypatch.setenv("GQH_DEV", gqh_dev)
    with pytest.raises(SystemExit):
        run_all.main(["--oos", "--databento-ok"])
    assert network_log == []


@pytest.mark.parametrize("gqh_dev", ["1", ""])
def test_every_test_window_download_needs_tag_and_clean_tree(gqh_dev, monkeypatch, before_tag, network_log):
    monkeypatch.setenv("GQH_DEV", gqh_dev)
    with pytest.raises(tl.GateError):
        oos_data.download(dest=before_tag / "d")
    with pytest.raises(tl.GateError):
        dbf.download_oos(approved=True)
    monkeypatch.setattr(tl, "_head_tags", lambda: [tl.GATE2_TAG])
    monkeypatch.setattr(tl, "_dirty", lambda: True)
    with pytest.raises(tl.GateError):
        oos_data.download(dest=before_tag / "d")
    with pytest.raises(tl.GateError):
        dbf.download_oos(approved=True)
    assert network_log == []


def test_download_guard_outside_git(monkeypatch):
    def fail():
        raise OSError("git not found")
    monkeypatch.setattr(tl, "_head_tags", fail)
    with pytest.raises(tl.GateError):
        tl.assert_gate2_download()


@pytest.mark.parametrize("where", ["OOS_LOG", "OOS_MARK"])
def test_second_run_needs_force_rerun(where, monkeypatch, before_tag, network_log):
    """A completed run (runs/oos_run.log) or one that stopped after its start (the git-ignored marker) is a run."""
    monkeypatch.setenv("GQH_DEV", "1")
    monkeypatch.setattr(oos_data, "committed_ok", lambda: True)
    getattr(run_all, where).write_text("2026-10-05T00:00:00Z RUN START commit abc dirty=False mode=download\n")
    with pytest.raises(SystemExit) as e:
        run_all.main(["--oos"])
    assert "--force-rerun" in str(e.value.code)
    assert network_log == []


def test_flags_need_oos():
    with pytest.raises(SystemExit) as e:
        run_all.main(["--databento-ok"])
    assert "--oos" in str(e.value.code)


def test_merge_oos_places_blocks():
    res = {"meta": {"pending": ["oos, flowclock.oos and H8.oos (Gate 2 only)", "other"]},
           "flowclock": {"in_sample": {}}, "H8": {"in_sample": {}}, "futures": {"status": "x"}, "figures": {}}
    oos = {"meta": {"commit": "abc"}, "oos": {"H1": {"b": 1.0}}, "flowclock": {"H6": {}}, "H8": {"c": {}},
           "futures": {"sample": []}, "figure": {"path": "p"}}
    run_all.merge_oos(res, oos)
    assert res["oos"] == oos["oos"] and res["flowclock"]["oos"] == oos["flowclock"]
    assert res["H8"]["oos"] == oos["H8"] and res["futures"]["oos"] == oos["futures"]
    assert res["figures"]["equity_curve_oos"] == {"path": "p"} and res["meta"]["pending"] == ["other"]


def test_in_sample_trials_drop_oos_rows():
    t = pd.DataFrame({"window": ["in_sample", "oos", "oos_futures_cost2x", "post_publication"]})
    assert tl.in_sample_trials(t)["window"].tolist() == ["in_sample", "post_publication"]


def test_labels():
    assert oos_eval.sign_label(0.1, 0.2) == "consistent"
    assert oos_eval.sign_label(-0.1, 0.2) == "not consistent"
    assert oos_eval.sign_label(0.0, 0.2) == "not consistent"
    assert oos_eval.sign_label(None, 0.2) == "not computed"
    oos = {"month_end": {"H1": {"b": -0.1}, "H4": {"diff": 0.2}}, "H8": {"c": {"b": 0.1, "p_one_sided": 0.2}}}
    lab = oos_eval.labels(oos, {}, {})
    assert lab["headline"]["H1"]["label"] == "sign not confirmed"
    assert lab["headline"]["H4"]["label"] == "pass"
    assert lab["headline"]["H8"]["label"] == "fail"
    kill = {k["condition"]: k["met"] for k in lab["kill_conditions"]}
    assert kill["b <= 0"] is True and kill["effect only in the 1990s"] is None


def test_splice_keeps_base_through_split_and_new_after(tmp_path):
    base = pd.DataFrame({"date": ["2024-09-27", "2024-09-30", "2024-10-01"], "v": ["a", "b", "OLD"]})
    new = pd.DataFrame({"date": ["2024-09-30", "2024-10-01", "2026-10-01"], "v": ["x", "c", "late"]})
    out = oos_data.splice(base, new, "date", "2024-09-30", "2026-09-30")
    assert out["v"].tolist() == ["a", "b", "c"]
    with pytest.raises(ValueError):
        oos_data.splice(base, new.rename(columns={"v": "w"}), "date", "2024-09-30", "2026-09-30")


def test_view_reads_like_the_snapshot_through_the_split(tmp_path):
    """A view whose 'downloaded' part is the snapshot's own rows after the split reads exactly like the snapshot
    (every file, as strings, through the end)."""
    split, end = "2022-09-30", "2024-09-30"
    new = tmp_path / "new"
    new.mkdir()
    for name, col in oos_data.DATE_COL.items():
        oos_data.write_csv(oos_data.rows_between(oos_data.read_str(oos_data.SNAPSHOT_DIR / name), col, split, end),
                           name, new)
    oos_data.build_view(new, split=split, end=end, view_dir=tmp_path / "view")
    for name, col in oos_data.DATE_COL.items():
        snap = oos_data.rows_between(oos_data.read_str(oos_data.SNAPSHOT_DIR / name), col, None, end)
        view = oos_data.read_str(tmp_path / "view" / name)
        assert view.reset_index(drop=True).equals(snap.reset_index(drop=True)), name


def test_pseudo_window_equals_reference_engines(monkeypatch):
    """`python run_all.py --oos-pseudo` (section 17): every daily series and statistic equal (<= 1e-12)."""
    monkeypatch.setenv("GQH_DEV", "")
    assert run_all.run_oos_pseudo() == 0


def test_results_oos_json_is_merged_when_present():
    """A plain run merges outputs/results_oos.json when it exists (section 17); the committed results.json then
    holds the same blocks."""
    if not run_all.OOS_JSON.exists():
        pytest.skip("no outputs/results_oos.json yet (before the Gate 2 run)")
    oos = json.loads(run_all.OOS_JSON.read_text(encoding="utf-8"))
    res = json.loads(run_all.report.RESULTS_JSON.read_text(encoding="utf-8"))
    assert res["oos"] == oos["oos"] and res["H8"]["oos"] == oos["H8"]


def test_oos_reproduce_path_end_to_end(monkeypatch, tmp_path):
    """The whole `--oos` path (keyless reproduction from a committed data/oos/) on the pseudo-window, with every
    output redirected to tmp_path: the run-once log, the data view, evaluate_window, labels, tables, figure,
    results_oos.json and its merge into results.json; then a second counted run refuses without --force-rerun.
    GQH_DEV=1 with the real Gate 2 date guard: HEAD counts as tagged, and the tree counts as dirty as soon as the run
    log or results_oos.json exists, so a file written into the repo before every block is computed fails the run."""
    split, start, end = "2022-09-30", "2022-10-01", "2024-09-30"
    monkeypatch.setenv("GQH_DEV", "1")
    monkeypatch.setattr(tl, "IS_END", split)            # the date guard treats the pseudo-window as the test window
    monkeypatch.setattr(tl, "_head_tags", lambda: [tl.GATE2_TAG])
    monkeypatch.setattr(tl, "_dirty", lambda: run_all.OOS_LOG.exists() or run_all.OOS_JSON.exists())
    for k, v in (("IS_END", split), ("OOS_START", start), ("OOS_END", end)):
        monkeypatch.setattr(run_all, k, v)
    oos_dir, out = tmp_path / "oos", tmp_path / "outputs"
    oos_dir.mkdir()
    for name, col in oos_data.DATE_COL.items():
        oos_data.write_csv(oos_data.rows_between(oos_data.read_str(oos_data.SNAPSHOT_DIR / name), col, split, end),
                           name, oos_dir)
    (oos_dir / oos_data.VINTAGE_NAME).write_text(json.dumps({"split": split, "end": end, "files": {}}))
    out.mkdir()
    (out / "results.json").write_text(run_all.report.RESULTS_JSON.read_text(encoding="utf-8"))
    monkeypatch.setattr(oos_data, "OOS_DATA_DIR", oos_dir)
    monkeypatch.setattr(oos_data, "VIEW_DIR", tmp_path / "view")
    monkeypatch.setattr(oos_data, "committed_ok", lambda: True)
    monkeypatch.setattr(dbf, "have_oos_cache", lambda: False)
    monkeypatch.setattr(run_all.report, "OUTPUTS", out)
    monkeypatch.setattr(run_all.report, "RESULTS_JSON", out / "results.json")
    monkeypatch.setattr(run_all, "OOS_JSON", out / "results_oos.json")
    monkeypatch.setattr(run_all, "FIG_DIR", out / "figures")
    monkeypatch.setattr(run_all, "OOS_LOG", tmp_path / "oos_run.log")
    monkeypatch.setattr(run_all, "OOS_MARK", tmp_path / "cache" / "run_started.log")
    logged = []
    monkeypatch.setattr(run_all, "log_trials", lambda rows: logged.extend(rows) or len(rows))
    run_all.main(["--oos"])
    oos = json.loads((out / "results_oos.json").read_text())
    res = json.loads((out / "results.json").read_text())
    assert res["oos"] == oos["oos"] and res["flowclock"]["oos"] == oos["flowclock"] and res["H8"]["oos"] == oos["H8"]
    assert res["futures"]["oos"]["status"].startswith("skipped")
    assert {k: v["label"] for k, v in oos["oos"]["labels"]["headline"].items()}.keys() == {"H1", "H4", "H8"}
    assert len(oos["oos"]["labels"]["tests"]) == len(oos_eval.CONSISTENCY_ROWS)
    assert oos["oos"]["sample"] == ["2022-10", "2024-09"]
    assert (out / "figures" / "equity_curve_oos.png").exists() and (out / "tables" / "windows_oos.csv").exists()
    assert [r["window"] for r in logged] == ["oos", "oos_placebo", "oos_cost2x", "oos_curve_allocated",
                                             "oos_curve_allocated_cost2x", "oos_flowclock", "oos_flowclock",
                                             "oos_flowclock_cost2x", "oos_flowclock_cost2x", "oos_h8"]
    log = (tmp_path / "oos_run.log").read_text().splitlines()
    assert " RUN START " in log[0] and " RUN COMPLETE " in log[1] and not run_all.OOS_MARK.exists()
    with pytest.raises(SystemExit) as e:
        run_all.main(["--oos"])
    assert "--force-rerun" in str(e.value.code)
