# GUI-JEPA attribution and protocol adaptation

This additive GUI decision experiment adapts data, candidate enumeration,
argument resolution and scoring code from **GUI-JEPA: Candidate-Conditioned JEPA
Latents as a Fast Action Policy for Mobile GUI Agents** by Bingchao Wang and the
GUI-JEPA authors. Its MIT copyright and permission notice are preserved in
[LICENSE](LICENSE). The original copyright applies to the adapted portions.

## Immutable source identity

- Repository: <https://github.com/bcwang-sjtu/GUI-JEPA>
- Commit: `dae79af4598962b9a5112b0bef297b588592192d`
- That commit's tree: `914b405017a061cb654607122124702f18ded5ce`
- Commit metadata: <https://api.github.com/repos/bcwang-sjtu/GUI-JEPA/commits/dae79af4598962b9a5112b0bef297b588592192d>
- Paper: <https://arxiv.org/abs/2610.36877>
- Published model: <https://huggingface.co/LilCC/GUI-JEPA>
- Model revision: `a2265ca26dbea66fc59c46240a90d3be5defdb9e`
- The model card records its original evaluation-code revision separately as
  `c03c565065a8b31f9d4ce1be6c29104c8869799f`.

The source commit was resolved using the official GitHub commit API. It is not
a guessed branch name or a tree hash presented as a commit. No model weights,
dataset shards or results are included in this directory.

## Adapted source map

All upstream paths below refer to the pinned commit above.

| Upstream source | Local adaptation |
| --- | --- |
| `gui_jepa/data/androidcontrol.py` | `src/improving/gui_decision/androidcontrol.py` |
| `scripts/preprocess/dump_candidate_texts.py` | Candidate label construction in `androidcontrol.py` |
| `gui_jepa/actions.py` | `src/improving/gui_decision/actions.py` |
| `scripts/preprocess/build_cache.py`, `dump_texts.py`, `export_screenshots.py` | Raw-data orchestration in `prepare.py`, without the ViT cache |
| `gui_jepa/text_utils.py` | Heuristics, span support and normalization in `arguments.py`; past-action history in `actions.py` |
| `gui_jepa/models.py`: `CrossEncoderMatcher`, `load_span_ranker` | Local-only frozen MiniLM-L6 span-ranker loading in `arguments.py` |
| `scripts/train_policy.py`: evaluation section | Canonical main-policy judge in `evaluation.py` |

The local data/model contracts, JSONL offset dataset, Qwen backbone integration,
output head, workflow and integrity checks are additions. This implementation
does not import the GUI-JEPA package, its ViT backbone, its predictor, or its
precomputed latent cache.

## Preserved protocol

- Candidate elements use the current accessibility forest, retain visible
  interactive nodes, deduplicate identical boxes, sort by ascending area and
  keep at most 80 by default. Current action labels never generate candidates.
- Every element offers click; editable elements also offer input_text. Four
  scroll directions and back/home/wait/open_app are global candidates.
- The recorded click is mapped to the smallest containing enumerated element.
  Its box supplies supervision only. Uncovered labels are retained with -100;
  every evaluation step stays in the denominator.
- Long press maps to click, including the upstream judge's accepted alias.
  input_text uses the upstream first-editable training convention; its native
  score checks typed text rather than the editable element's identity.
- Low has a step instruction; High has only the goal and allowed past history.
  Both are offline next-action prediction with recorded ground-truth history.
  They do not measure closed-loop task success.
- Input text is compared after lowercasing and whitespace normalization.
  Application names follow **the main GUI-JEPA judge's case-sensitive exact
  comparison**. The upstream Qwen reference normalizes app names, a separate
  rule deliberately not substituted here.
- The released span-ranker format is `ranker_lm/` (MiniLM-L6 and tokenizer) plus
  `ranker_head.pt` (a linear-head state dictionary). No generative text decoder
  is used for arguments in this resolver.

## Explicit implementation differences

- Raw PNG screenshot bytes and JSONL metadata replace frozen ViT feature
  tensors. Image resizing belongs to the Qwen processor and must be recorded
  by the experiment configuration. Actual PNG bytes are hashed when features
  are extracted; the feature-cache observation digest binds that image hash.
- TFRecord framing is strict and CRC32C verification is on by default. The
  explicit `--skip-crc-checks` option is recorded; truncation still fails.
- Preparation requires every episode in the supplied split files. Default
  malformed-record handling is fatal. `--on-error quarantine_train` can omit
  only malformed training episodes with detailed receipts; malformed or
  missing validation/test episodes remain fatal.
- Test memberships remain sets. `paper_idd` explicitly removes OOD overlaps
  from IDD, avoiding the upstream JSON-key overwrite mechanism. Full test is
  the default; `official_idd` and the OOD subsets are also available.
- The model receives a strict observation allowlist. Current labels, current
  action arguments and target boxes never enter its input. Argument resolution
  branches on the predicted kind before the independent judge sees the target.
- Span candidates use the same word n-gram/quoted-span support and limits, but
  use sorted unique spans and a deterministic source-text seed for the shuffle.
  This avoids the original set-order/global-RNG dependence and makes memoized
  repeated evaluation stable. Capped candidate membership or ties can differ;
  this is **not a claim of bit-identical published-score reproduction**. Use
  the same fixed resolver for compared policies and preserve `describe()`.
- Score reports separate all-click success, candidate coverage and the native
  click-grounding rate conditional on a correct predicted action type.

## Data and model terms

AndroidControl remains an external Google Research dataset at
<https://github.com/google-research/google-research/tree/master/android_control>
and `gs://gresearch/android_control`. Data/model terms are not replaced by this
MIT source-code license. MiniLM derivatives retain their upstream terms; model
downloads must retain the model bundle's notices and license files.

Current official split counts are train 13,603 / validation 137 / test 1,543.
The GUI-JEPA paper's stated 13,604 original training episodes differs by one;
its reported 13,594 cached training episodes are a separate author statistic.
Preparation records the actual supplied split IDs, output counts, quarantines,
source JSON hashes and prepared JSONL hashes instead of claiming those author
cache counts were reproduced.

Status of this source contribution: **generated_unexecuted**. Source review
does not constitute runtime qualification, native scorer parity testing or a
scientific result.
