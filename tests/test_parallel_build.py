"""Parallel profile builds: worker-count detection, and identical output to a serial build."""

import json

from textgrader import parallel
from textgrader.corpus import build_profile


def test_auto_jobs_takes_the_smallest_limit(monkeypatch):
    monkeypatch.setattr(parallel, "usable_cpus", lambda: 24)
    monkeypatch.setattr(parallel, "available_memory", lambda: 10 * 2**30)
    assert parallel.auto_jobs(100, per_worker=2.5 * 2**30)[0] == 4    # memory-bound
    assert parallel.auto_jobs(3, per_worker=2.5 * 2**30)[0] == 3      # file-bound
    monkeypatch.setattr(parallel, "usable_cpus", lambda: 1)
    assert parallel.auto_jobs(100)[0] == 1                            # one CPU
    monkeypatch.setattr(parallel, "available_memory", lambda: None)
    monkeypatch.setattr(parallel, "usable_cpus", lambda: 6)
    assert parallel.auto_jobs(100)[0] == 6                            # memory unknown


def test_explicit_jobs_are_bounded_by_the_file_count():
    assert parallel.resolve_jobs("8", 3)[0] == 3
    assert parallel.resolve_jobs(1, 50) == (1, "set by --jobs")


def test_cgroup_quota_caps_the_cpu_count(monkeypatch):
    monkeypatch.setattr(parallel, "_read", lambda path: "200000 100000"
                        if path == "/sys/fs/cgroup/cpu.max" else None)
    monkeypatch.setattr(parallel.os, "sched_getaffinity", lambda pid: set(range(16)),
                        raising=False)
    assert parallel.usable_cpus() == 2


def test_a_parallel_build_is_byte_identical_to_a_serial_one(tmp_path):
    for index in range(4):
        (tmp_path / f"book{index}.txt").write_text(
            " ".join(f"Sentence {index} number {n} walks to the old house and waits."
                     for n in range(120 + 40 * index)), encoding="utf-8")
    common = dict(corpus_name="t", built_at="2026-01-01T00:00:00Z", metrics={},
                  metric_selection="enabled")
    serial = build_profile([tmp_path], jobs=1, **common)
    parallel_profile = build_profile([tmp_path], jobs=2, **common)
    dump = lambda profile: json.dumps(profile, sort_keys=True)
    assert dump(parallel_profile) == dump(serial)
