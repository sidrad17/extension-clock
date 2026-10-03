"""Tests for src/trial_log.py guards: OOS dates refused without the gate2-frozen tag."""
import pytest

import src.trial_log as tl


@pytest.fixture
def dev(monkeypatch):
    monkeypatch.setattr(tl, "dev_mode", lambda: True)
    monkeypatch.setattr(tl, "_tags", lambda: [tl.GATE1_TAG])
    monkeypatch.setattr(tl, "_prereg_changed", lambda: False)
    monkeypatch.setattr(tl, "_head_tags", lambda: [])
    monkeypatch.setattr(tl, "_dirty", lambda: False)
    return monkeypatch


def test_oos_dates_refused_without_tag(dev):
    for end in ["2024-10-01", "2026-09-30", None]:
        with pytest.raises(tl.GateError):
            tl.guard_end(end)


def test_in_sample_dates_allowed(dev):
    tl.guard_end("2024-09-30")
    tl.guard_end("1999-12-31")


def test_oos_allowed_on_clean_frozen_head(dev):
    dev.setattr(tl, "_head_tags", lambda: [tl.GATE2_TAG])
    tl.guard_end("2026-09-30")
    dev.setattr(tl, "_dirty", lambda: True)
    with pytest.raises(tl.GateError):
        tl.guard_end("2026-09-30")


def test_gate1_requires_tag_and_unchanged_prereg(dev):
    tl.assert_gate1()
    dev.setattr(tl, "_prereg_changed", lambda: True)
    with pytest.raises(tl.GateError):
        tl.assert_gate1()
    dev.setattr(tl, "_tags", lambda: [])
    with pytest.raises(tl.GateError):
        tl.assert_gate1()


def test_guards_off_without_gqh_dev(monkeypatch):
    monkeypatch.setattr(tl, "dev_mode", lambda: False)
    monkeypatch.setattr(tl, "_head_tags", lambda: [])
    tl.guard_end("2026-09-30")
    tl.assert_gate1()


def test_dev_mode_reads_environment(monkeypatch):
    monkeypatch.setenv("GQH_DEV", "1")
    assert tl.dev_mode()
    monkeypatch.setenv("GQH_DEV", "0")
    assert not tl.dev_mode()
