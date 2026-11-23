"""Explicit command surface for the Qwen GUI dual-head migration."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .cache import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/gui_decision/qwen35_08b_2060.json")
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download", help="Acquire/verify pinned inputs, without running models")
    download.add_argument("--asset", choices=("all", "qwen", "arguments", "data"), default="all")
    download.add_argument("--verify-only", action="store_true")
    prepare = commands.add_parser("prepare", help="Raw AndroidControl -> PNG/JSONL; no emulator")
    prepare.add_argument("--raw-dir")
    prepare.add_argument("--output-dir")
    prepare.add_argument("--splits")
    prepare.add_argument("--test-subsplits")
    prepare.add_argument("--on-error", choices=("error", "quarantine_train"), default="error")
    extract = commands.add_parser("extract", help="Cache frozen Qwen ACT features for every requested view")
    extract.add_argument("--split", choices=("train", "validation", "test"), required=True)
    extract.add_argument("--mode", choices=("low", "high", "both"), default="both")
    extract.add_argument("--subset", default="full")
    extract.add_argument("--output-dir")
    extract.add_argument("--resume", action="store_true")
    train = commands.add_parser("train", help="Train only the decision head using cached train features")
    train.add_argument("--output-dir")
    train.add_argument("--resume", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Native ACT-only step metrics using cached features")
    evaluate.add_argument("--checkpoint")
    evaluate.add_argument("--split", choices=("validation", "test"), default="test")
    evaluate.add_argument("--output-dir")
    online = commands.add_parser("evaluate-online", help="Actual shared-model ACT or PLAN_ACT inference")
    online.add_argument("--checkpoint")
    online.add_argument("--split", choices=("validation", "test"), default="test")
    online.add_argument("--mode", choices=("low", "high"), default="high")
    online.add_argument("--route", choices=("ACT", "PLAN_ACT"), default="PLAN_ACT")
    online.add_argument("--subset", default="full")
    online.add_argument("--output-dir")
    predict = commands.add_parser("predict", help="One observation; no supervision accepted")
    predict.add_argument("--checkpoint")
    predict.add_argument("--observation", required=True)
    predict.add_argument("--gate-prompt", default="<GUI_MODE:PLAN_ACT>")
    predict.add_argument("--output", required=True)
    export = commands.add_parser("export-observation", help="Export a sanitized observation for one-step inference")
    export.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    export.add_argument("--mode", choices=("low", "high"), default="high")
    export.add_argument("--index", type=int, default=0)
    export.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    config = read_json(args.config)
    checkpoint = getattr(args, "checkpoint", None) or str(Path(config["training"]["output_dir"]) / "best")
    if args.command == "download":
        from .assets import acquire
        result = acquire(config, config["assets"]["pins"], args.asset, args.verify_only)
    elif args.command == "prepare":
        from .prepare import main as prepare_main
        values = ["--raw-dir", args.raw_dir or config["data"]["raw_dir"],
                  "--output-dir", args.output_dir or config["data"]["prepared_dir"],
                  "--on-error", args.on_error]
        if args.splits:
            values += ["--splits", args.splits]
        if args.test_subsplits:
            values += ["--test-subsplits", args.test_subsplits]
        return prepare_main(values)
    elif args.command == "extract":
        from .runner import extract as run_extract
        result = run_extract(config, args.split, args.mode, args.subset, args.output_dir, args.resume)
    elif args.command == "train":
        from .runner import train as run_train
        result = run_train(config, args.output_dir, args.resume)
    elif args.command == "evaluate":
        from .runner import evaluate as run_evaluate
        result = run_evaluate(config, checkpoint, args.split, args.output_dir)
    elif args.command == "evaluate-online":
        from .runner import evaluate_online
        result = evaluate_online(config, checkpoint, args.split, args.mode, args.route, args.subset, args.output_dir)
    elif args.command == "predict":
        from .runner import predict as run_predict
        result = run_predict(config, checkpoint, read_json(args.observation), args.gate_prompt)
        write_json(args.output, result)
    else:
        from .dataset import GuiDecisionDataset
        dataset = GuiDecisionDataset(config["data"]["prepared_dir"], split=args.split, mode=args.mode)
        result = dataset[args.index]["observation"]
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
