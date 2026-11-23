"""Target-free action-argument resolution with an optional frozen MiniLM.

Native heuristics and the released span-ranker format are adapted from GUI-JEPA
(MIT), pinned in third_party/gui_jepa. No weights are downloaded here. Span
sampling is made deterministic per source text rather than depending on Python
set order and the global RNG; this is recorded by describe(), not presented as
bit-identical reproduction of the original inference implementation.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

from .actions import prediction_from_candidate
from .schema import GUI_JEPA_MODEL_REVISION, validate_model_observation


def extract_type_text(instruction: str, goal: str) -> str:
    for source in (instruction, goal):
        if not source:
            continue
        quoted = re.search(r"['\"]([^'\"]+)['\"]", source)
        if quoted:
            return quoted.group(1)
        match = re.search(r"(?:type|enter|input|search for|search|write|fill in|fill)\s+(?:in\s+)?(.+?)(?:\s+(?:in|into|on|at)\s+the\b.*)?$",
                          source, re.IGNORECASE)
        if match:
            return match.group(1).strip(" .")
    return (instruction or goal).strip(" .")


def predict_app(instruction: str, goal: str, vocabulary: Sequence[str]) -> str:
    source = f" {instruction} {goal} ".lower()
    best = None
    for name in vocabulary:
        if name.lower() in source and (best is None or len(name) > len(best)):
            best = name
    return best or (vocabulary[0] if vocabulary else "")


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def span_candidates(source: str, *, max_words: int = 16, cap: int = 800,
                    seed: int = 1) -> list[str]:
    """Native n-gram/quoted-span support with stable source-keyed sampling."""
    if max_words <= 0 or cap <= 0:
        raise ValueError("max_words and cap must be positive")
    source = (source or "").strip()
    words = source.split()
    spans = {" ".join(words[i:i + n]).strip(" .,'\"") for i in range(len(words))
             for n in range(1, min(max_words, len(words) - i) + 1)}
    # Sorting removes PYTHONHASHSEED dependence; source hashing avoids changes
    # when evaluation order, Low/High interleaving, or cache hits change.
    result = sorted(spans)
    source_seed = int.from_bytes(hashlib.sha256(f"{seed}\0{source}".encode("utf-8")).digest()[:8], "big")
    random.Random(source_seed).shuffle(result)
    quoted = re.search(r"['\"]([^'\"]+)['\"]", source)
    if quoted:
        result = [quoted.group(1)] + result
    return result[:cap]


def load_app_vocab(value: str | Path | Sequence[str]) -> list[str]:
    if isinstance(value, (str, Path)):
        vocabulary = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        vocabulary = list(value)
    if not isinstance(vocabulary, list) or any(not isinstance(name, str) for name in vocabulary):
        raise ValueError("Application vocabulary must be an ordered JSON list of strings")
    if len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Application vocabulary contains duplicates")
    return vocabulary


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class _FrozenSpanRanker:
    """The published MiniLM-L6 + one-unit Linear head, loaded locally only."""

    def __init__(self, directory: Path, device: str, batch_size: int) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device, self.batch_size = torch.device(device), batch_size
        lm_path = directory / "ranker_lm"
        self.tokenizer = AutoTokenizer.from_pretrained(str(lm_path), local_files_only=True)
        self.lm = AutoModel.from_pretrained(str(lm_path), local_files_only=True,
                                          torch_dtype=torch.float32).to(self.device).eval()
        self.head = torch.nn.Linear(self.lm.config.hidden_size, 1).to(self.device).eval()
        state = torch.load(directory / "ranker_head.pt", map_location="cpu", weights_only=True)
        self.head.load_state_dict(state, strict=True)
        for module in (self.lm, self.head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def score(self, source: str, spans: Sequence[str]) -> list[float]:
        scores = []
        with self.torch.inference_mode():
            for start in range(0, len(spans), self.batch_size):
                chunk = spans[start:start + self.batch_size]
                encoded = self.tokenizer([source[:300]] * len(chunk),
                                         [span or "[empty]" for span in chunk],
                                         padding=True, truncation=True, max_length=96,
                                         return_tensors="pt").to(self.device)
                hidden = self.lm(**encoded).last_hidden_state[:, 0]
                scores.extend(self.head(hidden).squeeze(-1).float().cpu().tolist())
        return scores


class ArgumentResolver:
    """Resolve from the predicted kind and sanitized input, never current GT.

    Default full-system mode requires the released span ranker. The explicit
    heuristic mode is available for a distinct ablation, never a fallback on a
    missing checkpoint. Repeated sources are memoized without any label keys.
    """

    def __init__(self, *, app_vocab: str | Path | Sequence[str], mode: str = "span_ranker",
                 span_checkpoint: str | Path | None = None, device: str = "cpu",
                 max_span_words: int = 16, span_cap: int = 800, span_seed: int = 1,
                 span_batch_size: int = 64) -> None:
        if mode not in ("span_ranker", "heuristic"):
            raise ValueError("Argument mode must be span_ranker or heuristic")
        if min(max_span_words, span_cap, span_batch_size) <= 0:
            raise ValueError("Span limits and batch size must be positive")
        self.mode, self.device = mode, device
        self.vocabulary = load_app_vocab(app_vocab)
        self.vocab_source = str(Path(app_vocab).resolve()) if isinstance(app_vocab, (str, Path)) else "provided_ordered_list"
        self.max_span_words, self.span_cap = max_span_words, span_cap
        self.span_seed, self.span_batch_size = span_seed, span_batch_size
        self.span_checkpoint: Path | None = None
        self.span_checkpoint_files: dict[str, str] = {}
        if mode == "span_ranker":
            if span_checkpoint is None:
                raise ValueError("Full-system evaluation requires --span-checkpoint; heuristic mode must be explicit")
            directory = Path(span_checkpoint).resolve()
            if (directory / "span_ranker").is_dir():
                directory = directory / "span_ranker"
            if not (directory / "ranker_head.pt").is_file() or not (directory / "ranker_lm").is_dir():
                raise FileNotFoundError(f"Missing released ranker_head.pt / ranker_lm under {directory}")
            lm_directory = directory / "ranker_lm"
            if not (lm_directory / "config.json").is_file():
                raise FileNotFoundError(f"Missing ranker configuration under {lm_directory}")
            if not any((lm_directory / name).is_file() for name in ("model.safetensors", "pytorch_model.bin")):
                raise FileNotFoundError(f"Missing ranker model weights under {lm_directory}")
            if not any((lm_directory / name).is_file() for name in ("tokenizer.json", "vocab.txt")):
                raise FileNotFoundError(f"Missing local ranker tokenizer under {lm_directory}")
            self.span_checkpoint = directory
            # Bind the actual content rather than only a mutable directory or
            # an asserted upstream revision. This reads bytes, not model code.
            self.span_checkpoint_files = {
                path.relative_to(directory).as_posix(): _file_sha256(path)
                for path in sorted(directory.rglob("*")) if path.is_file()
            }
        self._ranker: _FrozenSpanRanker | None = None
        self._cached_text = lru_cache(maxsize=8192)(self._resolve_text)
        self._cached_app = lru_cache(maxsize=8192)(self._resolve_app)

    def _resolve_text(self, instruction: str, goal: str) -> str:
        if self.mode == "heuristic":
            return extract_type_text(instruction, goal)
        source = instruction or goal
        spans = span_candidates(source, max_words=self.max_span_words, cap=self.span_cap, seed=self.span_seed)
        if not spans:
            return extract_type_text(instruction, goal)
        if self._ranker is None:
            assert self.span_checkpoint is not None
            self._ranker = _FrozenSpanRanker(self.span_checkpoint, self.device, self.span_batch_size)
        scores = self._ranker.score(source, spans)
        if len(scores) != len(spans):
            raise RuntimeError("Span ranker returned an incompatible score count")
        return spans[max(range(len(scores)), key=scores.__getitem__)]

    def _resolve_app(self, instruction: str, goal: str) -> str:
        return predict_app(instruction, goal, self.vocabulary)

    def resolve(self, candidate: Mapping[str, Any], observation: Mapping[str, Any]) -> dict[str, Any]:
        validate_model_observation(observation)
        if not any(dict(candidate) == allowed for allowed in observation["candidates"]):
            raise ValueError("Selected action is absent from the observation's candidate list")
        prediction = prediction_from_candidate(candidate)
        # This branch is based exclusively on the *predicted* kind.
        if prediction["kind"] == "input_text":
            prediction["text"] = self._cached_text(observation["instruction"], observation["goal"])
        elif prediction["kind"] == "open_app":
            prediction["app_name"] = self._cached_app(observation["instruction"], observation["goal"])
        return prediction

    def describe(self) -> dict[str, Any]:
        return {
            "mode": self.mode, "app_vocab_source": self.vocab_source,
            "app_vocab_size": len(self.vocabulary),
            "app_vocab_sha256": hashlib.sha256(json.dumps(self.vocabulary, ensure_ascii=False,
                                                          separators=(",", ":")).encode("utf-8")).hexdigest(),
            "span_checkpoint": str(self.span_checkpoint) if self.span_checkpoint else None,
            "span_checkpoint_files_sha256": dict(self.span_checkpoint_files),
            "span_checkpoint_fingerprint": hashlib.sha256(json.dumps(self.span_checkpoint_files,
                                                                       sort_keys=True).encode("utf-8")).hexdigest(),
            "released_repo": "LilCC/GUI-JEPA", "released_revision": GUI_JEPA_MODEL_REVISION,
            "span_checkpoint_content_verified_by_this_class": False,
            "device": self.device, "max_span_words": self.max_span_words, "span_cap": self.span_cap,
            "span_seed": self.span_seed, "span_batch_size": self.span_batch_size,
            "span_sampling": "sorted_unique_spans_then_source_seeded_shuffle",
            "argument_cache": "8192 instruction/goal pairs, no labels", "prediction_conditioning": "predicted_kind_only",
        }

