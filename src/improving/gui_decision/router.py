"""Explicit prompt routing of a shared Qwen backbone's two output heads.

This initial router is a deterministic protocol, not a learned difficulty gate.
PLAN generates text using the original vocabulary head. ACT scores candidates
using the pretrained decision head. PLAN_ACT composes both on the same model.
"""
from __future__ import annotations

import copy
import re
import time

from .schema import validate_model_observation

ROUTES = ("PLAN", "ACT", "PLAN_ACT")
ROUTER_VERSION = "explicit_prompt_dual_head_v1"


def parse_gate_prompt(gate_prompt: str) -> str:
    """Only the caller's control prompt selects the route, never screen text."""
    if not isinstance(gate_prompt, str):
        raise TypeError("gate_prompt must be a string")
    match = re.fullmatch(r"\s*<GUI_MODE:(PLAN|ACT|PLAN_ACT)>\s*", gate_prompt)
    if match is None:
        raise ValueError("Use exactly <GUI_MODE:PLAN>, <GUI_MODE:ACT>, or <GUI_MODE:PLAN_ACT>")
    return match.group(1)


def planning_observation(observation: dict) -> dict:
    validate_model_observation(observation)
    clean = copy.deepcopy(observation)
    # Autonomous planning never receives the oracle Low step instruction.
    clean["instruction"] = clean["goal"]
    return clean


class PromptGatedPolicy:
    def __init__(self, model, processor, resolver, *, max_plan_tokens=64):
        if max_plan_tokens < 1:
            raise ValueError("max_plan_tokens must be positive")
        self.model, self.processor, self.resolver = model, processor, resolver
        self.max_plan_tokens = max_plan_tokens
        self.model.eval()

    def plan(self, observation: dict) -> str:
        clean = planning_observation(observation)
        batch = self.processor.prepare_plan([clean])
        generated = self.model.generate_plan(batch["model_inputs"],
                                             max_new_tokens=self.max_plan_tokens)
        text = self.processor.processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not text:
            raise ValueError("Planning head produced an empty next-step instruction")
        return text

    def act(self, observation: dict) -> dict:
        import torch
        validate_model_observation(observation)
        with torch.no_grad():
            logits = self.model(self.processor([observation]))
        choice = int(logits.argmax(-1).item())
        selected = observation["candidates"][choice]
        prediction = self.resolver.resolve(selected, observation)
        return {"choice": choice, "candidate_id": selected["candidate_id"],
                "action": prediction, "scores": logits[0].float().cpu().tolist()}

    def run(self, observation: dict, gate_prompt="<GUI_MODE:PLAN_ACT>") -> dict:
        route = parse_gate_prompt(gate_prompt)
        validate_model_observation(observation)
        start = time.monotonic()
        result = {"sample_id": observation["sample_id"], "route": route,
                  "router_version": ROUTER_VERSION, "gate_kind": "explicit_prompt",
                  "plan": None, "decision_instruction_source": None}
        if route in ("PLAN", "PLAN_ACT"):
            result["plan"] = self.plan(observation)
        if route == "PLAN_ACT":
            act_observation = planning_observation(observation)
            act_observation["instruction"] = result["plan"]
            result.update(self.act(act_observation))
            result["decision_instruction_source"] = "same_model_generated_plan"
        elif route == "ACT":
            result.update(self.act(observation))
            result["decision_instruction_source"] = "caller_observation"
        result["elapsed_seconds"] = time.monotonic() - start
        return result
