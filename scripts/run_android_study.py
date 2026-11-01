#!/usr/bin/env python3
"""Entry point for the isolated AndroidWorld experiment."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/android/ui_voyager_qwen4b.json")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("plan")
    fetch = sub.add_parser("fetch")
    fetch.add_argument("--source-only", action="store_true")
    sub.add_parser("prepare")
    sub.add_parser("setup-env")
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--arm", choices=["feedback_spectrum", "feedback_spectrum_tau", "feedback_spectrum_gated"],
                         default="feedback_spectrum")
    sub.add_parser("warmup")
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=["both", "all", "ui_voyager", "feedback_spectrum",
                                       "feedback_spectrum_tau", "feedback_spectrum_gated"], default="both")
    sub.add_parser("report")
    args = parser.parse_args()
    from improving.android.study import FEEDBACK_ARMS, arm_config, configured_arms, load_config, run_study, report
    config = load_config(args.config)
    if args.command == "plan":
        tau_policies = {}
        for arm in configured_arms(config):
            if arm in FEEDBACK_ARMS:
                spectral = arm_config(config, arm)["spectral"]
                tau_policies[arm] = {"tau": spectral.get("tau", 1.), "tau_decay": spectral.get("tau_decay"),
                                     "tau_mode": spectral.get("tau_mode"), "tau_gamma": spectral.get("tau_gamma", 5.)}
        result = {"model": config["model"]["repo_id"], "arms": list(configured_arms(config)),
                  "tau_policies": tau_policies,
                  "shared_rft_rounds": config["study"]["warmup_rounds"], "refinement_rounds": config["study"]["rounds"],
                  "rollouts_per_train_instance_per_round": config["study"]["group_size"],
                  "output_dir": config["output_dir"], "data": config["data"],
                  "status": "plan_only", "no_grsd_in_our_arm": True}
    elif args.command == "fetch":
        from improving.android.assets import acquire
        result = acquire(config, source_only=args.source_only)
    elif args.command == "prepare":
        from improving.android.assets import verify_upstream
        from improving.android.environment import prepare_manifest
        verify_upstream(config["android"]["upstream_root"])
        result = prepare_manifest(config["android"]["upstream_root"], config["manifest"], config)
        # Private parameters and randomization remain on disk, not terminal logs.
        result = {"manifest": config["manifest"], "instances": len(result["instances"]), "status": "prepared"}
    elif args.command == "report":
        result = report(config)
    elif args.command == "setup-env":
        from improving.android.environment import _load_components, _open_env
        _, launcher, _ = _load_components(config["android"]["upstream_root"])
        settings = dict(config["android"], emulator_setup=True)
        env = _open_env(launcher, settings)
        try:
            env.get_state(wait_to_stabilize=True)
        finally:
            env.close()
        result = {"status": "native_environment_setup_complete"}
    elif args.command == "qualify":
        from improving.android.qualification import qualify
        result = qualify(config, args.arm)
    else:
        result = run_study(config, getattr(args, "arm", "all"), warmup_only=args.command == "warmup")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
