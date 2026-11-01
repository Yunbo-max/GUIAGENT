# GUI tau/gated integration handoff

Target: `Yunbo-max/SPECTRUM`, **android-feedback-spectrum** only; no main merge.
GUI parent: `eb0008b0fa64332803e75937e6df8ef07a209c9e`.
Existing algorithm source: main `b8bcc68f87a2a216ac048bc7fb36365c2002e8b0`.
Current entry: [AGENTS](../../AGENTS.md), [Android guide](../../docs/ANDROID_UI_VOYAGER_ZH.md),
[variant definitions, exact commands and acceptance](../../docs/ANDROID_VARIANTS_ZH.md).

Status: **generated_unexecuted**. This is the user's requested port of two existing
strength controls into the existing GUI experiment, not a new method-discovery claim.
Reuse model/upstream/manifest pins. New four-arm config has a fresh output root;
shared RFT and fixed-anchor acquisition are reused within it, while recursive
models, feedback and round-one entropy references are arm-local.

Only strength control changes: latest existing decay0.6 and adaptive gamma5.
Gate entropy uses current fixed-reference NLL moments; the intervention still
uses the GUI's existing anchor/feedback mixture. The reference is bound to the
first post-RFT refinement geometry receipt, including resumed execution.
Invalid adaptive state fails explicitly. Original entropy/rank normalization and
absence of clipping are documented, not recast as proven collapse prevention.

Local must run the authored regression sources, verify four-arm plan/receipt
semantics and qualify actual GPU/emulator resources before dispatch. No tests,
training, inference, downloads or native scorer run were executed by Web in this
revision. Earlier82 CPU passes belong to the old revision. Preserve all original
results and running checkouts. Capture the delivered commit from the readback
receipt; this parent/source record is not the new code's execution identity.

The complete requested comparison remains one4B model and AndroidWorld, shared
three-round RFT, then three rounds per arm, native verifier and independent frozen
evaluation instances. More arms increase total cost; no deployment or paid run
was started by source publication. Return the guide's per-round raw evidence,
tau/entropy histories, environment/update costs and unresolved statuses.
