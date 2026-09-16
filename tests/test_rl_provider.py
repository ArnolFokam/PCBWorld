"""Tests for methods/rl_agent/policy/rl_provider.py (the visualizer's local RL provider).

No checkpoint or model: ``KiCadRLAgent.from_checkpoint`` is replaced by a fake agent that
returns a fixed action, and the obs is the pure-Python mock from ``tests/_mock_obs.py``.

Regression: a checkpoint whose ``train_args`` lack ``policy_net_select`` must resolve it to
False (the training default) when building the action envelope — a policy trained with
wrapper-side net picking must not have its pointer read as a net index.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from methods.rl_agent.policy import agent as agent_mod
from methods.rl_agent.policy.rl_provider import RLPolicyProvider
from pcb_world.core.action_schema import ACT_NET_SELECT
from tests._mock_obs import make_mock_obs


class _FakeAgent:
    """Stands in for KiCadRLAgent: always emits net_select with pointer 1, walkaround."""

    def __init__(self, ckpt_args: dict):
        self.model = object()
        self.device = torch.device("cpu")
        self.train_iteration = 0
        self.ckpt_args = dict(ckpt_args)

    def act(self, obs_list, **kwargs):
        return torch.tensor([[ACT_NET_SELECT, 1, 2]]), torch.zeros(1)


@pytest.fixture
def make_provider(monkeypatch, tmp_path):
    def _make(train_args: dict) -> RLPolicyProvider:
        monkeypatch.setattr(
            agent_mod.KiCadRLAgent, "from_checkpoint",
            classmethod(lambda cls, path, device="auto", *, deterministic=False:
                        _FakeAgent(train_args)),
        )
        ckpt = tmp_path / "policy_best.pt"
        ckpt.write_bytes(b"")
        return RLPolicyProvider(SimpleNamespace(
            checkpoint_path=str(ckpt), device="cpu", temperature=0.0,
        ))
    return _make


@pytest.mark.parametrize("train_args, expected_body", [
    ({}, "net_select 1"),                          # key absent -> False -> first net, pointer ignored
    ({"policy_net_select": False}, "net_select 1"),
    ({"policy_net_select": True}, "net_select 2"),  # honoured -> pointer 1 indexes sorted codes
])
def test_policy_net_select_defaults_to_false_when_train_args_lack_it(
    make_provider, train_args, expected_body,
):
    provider = make_provider(train_args)
    assert provider.policy_net_select is bool(train_args.get("policy_net_select", False))

    obs = make_mock_obs()   # nets 1 and 2 -> sorted_net_codes == [1, 2]
    responses, token_counts = provider.generate([("", "")], observations=[obs])

    assert len(responses) == len(token_counts) == 1
    assert f"<action>{expected_body}</action>" in responses[0]
