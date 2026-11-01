"""Small native-host acceptance run; never a benchmark result."""
from pathlib import Path

from .assets import atomic_json, resolve_checkpoint, verify_upstream


def qualify(config: dict, arm: str = "feedback_spectrum") -> dict:
    """Exercise actual images, environment, K/V, SFT, KL and merged reload.

    Uses a separate output directory and only training/calibration instances.
    If real successes are absent this stops with an explicit signal shortage,
    rather than fabricating a reference or accepting a dummy model test.
    """
    from .study import (FEEDBACK_ARMS, LEGACY_ARMS, _collect, _instances, arm_config,
                        configured_arms, load_manifest, stage_config, study_lock,
                        validate_config, validate_geometry_tau)
    validate_config(config)
    settings = arm_config(config, arm)
    if arm not in FEEDBACK_ARMS:
        raise ValueError("Qualification requires a feedback SPECTRUM arm")
    import torch
    from .data import success_examples, feedback_examples
    from .model import QwenPolicy
    from .geometry import collect_operators
    from .training import train_sft, train_feedback
    from ..spectral import folded_operators

    verify_upstream(config["android"]["upstream_root"])
    manifest = load_manifest(config["manifest"])
    checkpoint = resolve_checkpoint(config)
    settings["runtime_seed"] = manifest["seed"]
    settings["train"]["max_updates"] = 1
    settings["train"]["gradient_accumulation_steps"] = 1
    settings["spectral"]["max_anchor_examples"] = 2
    settings["spectral"]["max_feedback_examples"] = 2
    root = Path(config["output_dir"]) / "qualification"
    if any(method not in LEGACY_ARMS for method in configured_arms(config)):
        root = root / arm
    with study_lock(root):
        if (root / "qualified.json").exists():
            raise FileExistsError("Qualification already exists; inspect its receipt rather than overwrite it")
        train_episodes = _collect(checkpoint, _instances(manifest, "train")[:2], root / "train_collection", settings, 2)
        calibration = _collect(checkpoint, _instances(manifest, "calibration")[:2], root / "calibration_collection", settings, 2)
        successes = success_examples(train_episodes)
        anchors = success_examples(calibration, split="calibration")
        feedback, evidence = feedback_examples(train_episodes, settings)
        if not successes or not anchors or not feedback:
            atomic_json(root / "signal_shortage.json", {"status": "not_qualified", "successful_train_steps": len(successes),
                        "anchor_steps": len(anchors), "feedback_steps": len(feedback),
                        "reason": "This bounded probe did not obtain enough real supervision. No fake labels were inserted."})
            raise RuntimeError(f"Qualification signal shortage; see {root / 'signal_shortage.json'}")
        policy = QwenPolicy(checkpoint, settings)
        try:
            operators, calibration_receipt = collect_operators(policy, anchors, feedback, settings, root / "geometry",
                                                               round_index=1, reference_entropy=None)
            validate_geometry_tau(calibration_receipt, settings["spectral"], 1)
            originals = {name: policy.model.get_submodule(name).weight.detach().cpu().clone() for name in operators}
            with folded_operators(policy.model, operators):
                sampled = policy.generate(feedback[0]["messages"])
            for name, original in originals.items():
                if not torch.equal(original, policy.model.get_submodule(name).weight.detach().cpu()):
                    raise RuntimeError("Native K/V weights were not exactly restored")
            if not sampled["response_token_ids"]:
                raise RuntimeError("Spectral generation returned no target tokens")
        finally:
            policy.close()
        sft = train_sft(checkpoint, successes[:1], root / "sft_checkpoint", stage_config(settings, "qualification_sft"))
        kl = train_feedback(checkpoint, feedback[:1], root / "feedback_checkpoint", stage_config(settings, "qualification_kl"))
        reloaded = QwenPolicy(str(root / "feedback_checkpoint"), settings)
        try:
            output = reloaded.generate(feedback[0]["messages"])
            if not output["response_token_ids"]:
                raise RuntimeError("Reloaded merged checkpoint returned no continuation")
        finally:
            reloaded.close()
        result = {"status": "native_probe_passed", "arm": arm, "geometry_round": 1,
                  "sft": sft, "feedback": kl, "calibration": calibration_receipt,
                  "feedback_evidence": evidence, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  "limitation": "Bounded first-round probe of this arm only; not full benchmark, long-context capacity, late-round gate behavior, or measured method improvement."}
        atomic_json(root / "qualified.json", result)
        return result
