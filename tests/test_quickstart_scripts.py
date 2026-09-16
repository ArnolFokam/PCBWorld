"""The README's runnable surface stays runnable.

The README names three quick start scripts (``tools/quickstart/{prepare_pcbench,train_rl,run_llm}.sh``) and shows one
Python example. Their real runs are the release clean-room's job; here only the contract every
reader hits first is pinned: each script parses, answers ``--help`` with its own usage line, and
refuses to run without ``PCBWORLD_DATA_ROOT`` when it needs it, naming the variable. The README
example is executed as written (first ``python`` block) when the engine is importable.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = sorted(p.relative_to(REPO) for p in (REPO / "tools" / "quickstart").glob("*.sh"))
NEEDS_DATA_ROOT = {"prepare_pcbench.sh", "train_rl.sh"}


def _run(args, env=None, timeout=60):
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=timeout, env=env)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.name)
def test_script_parses_and_documents_itself(script):
    assert _run(["bash", "-n", str(script)]).returncode == 0
    proc = _run(["bash", str(script), "--help"])
    assert proc.returncode == 0, proc.stderr
    assert f"bash {script}" in proc.stdout, "--help must show the README invocation"
    bad = _run(["bash", str(script), "--no-such-flag"])
    assert bad.returncode == 2 and "--help" in bad.stderr


@pytest.mark.parametrize("script", [s for s in SCRIPTS if s.name in NEEDS_DATA_ROOT], ids=lambda p: p.name)
def test_script_names_the_missing_data_root(script, monkeypatch):
    import os
    env = {k: v for k, v in os.environ.items() if k != "PCBWORLD_DATA_ROOT"}
    proc = _run(["bash", str(script)], env=env)
    assert proc.returncode != 0 and "PCBWORLD_DATA_ROOT" in proc.stderr


def test_readme_example_runs_as_written():
    pytest.importorskip("kicad_rl_router", reason="kicad_rl_router not available")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"```python\n(.*?)```", readme, re.S)
    assert m, "README has no python example"
    code = m.group(1) + "\nassert reward != 0 or terminated is not None\nprint('README-EXAMPLE-OK', reward)\n"
    proc = _run([sys.executable, "-c", code], timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "README-EXAMPLE-OK" in proc.stdout


def test_train_rl_evaluates_a_checkpoint_every_run_writes():
    """train_rl.sh evaluates ``policy_last.pt``: the trainer overwrites it on every iteration, while
    ``policy_best.pt`` appears only after an inline eval, which a few-iteration run never reaches."""
    script = (REPO / "tools" / "quickstart" / "train_rl.sh").read_text(encoding="utf-8")
    m = re.search(r'^CKPT="\$OUT/ppo_per_step/checkpoints/([^"]+)"$', script, re.M)
    assert m, "train_rl.sh no longer sets CKPT the way this test reads it"
    assert m.group(1) == "policy_last.pt"
    loop = (REPO / "methods" / "rl_agent" / "training" / "loop.py").read_text(encoding="utf-8")
    hook = re.search(r"def on_iteration_end\(self.*?\n(.*?)\n    def ", loop, re.S)
    assert hook and "self.save_last_ckpt(" in hook.group(1), "policy_last.pt is no longer written every iteration"
    assert '"policy_last.pt"' in loop
