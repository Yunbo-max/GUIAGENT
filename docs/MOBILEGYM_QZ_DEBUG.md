# MobileGym QZ Runbook & alipay Debug Notes

## Status

- **8/8/1 pilot (train_task_limit=4)**: end-to-end run succeeded, `report.json`
  generated. Both arms ran (feedback_sd + feedback_spectrum_learned); learned arm
  reports `learned_gain_status: "learned"`, `geometry_source: "feedback_only"`.
  Success rate is 0% (expected: no RFT warmup, 4 train tasks, 1 round).
- **alipay.SendMessageToContact**: crashes only on the qz bare-CUDA image
  (`relay2:v2`); runs fine on a machine with full desktop chromium libraries.

## Why alipay fails on qz

Diagnosis (`diag_gpu16` on qz):

```
初始 apps: [..., 'alipay', ...]          # alipay store IS registered initially
reset ok
open_app ok
get_state[0]: apps数=0, alipay=False     # after reset+open_app the store is GONE
```

Root cause chain:

1. `env.reset()` reloads the page; MobileGym app stores are lazily created after
   reload (their Zustand `storeRegistry.set(appId, store)` runs as a module side
   effect when the app's `state.ts` is imported via React.lazy).
2. `open_app()` (used by single-app tasks) does NOT preload the store — it only
   navigates. `preloadAppStores` is a no-op (`os/OSContext.tsx:686` "eager loaded").
3. On a full-desktop chromium, the alipay chunk loads fast enough that `get_state`
   still sees it; on qz's stripped chromium (LD_LIBRARY_PATH shimmed libs), the
   store never re-registers, so `get_state(required_apps=['alipay'])` fails.
4. `_setup_with_retry` then re-runs setup → `reset` navigates again → the qz
   chromium has already died → `TargetClosedError: browser has been closed`.

## Fixes already applied (committed)

- extended `get_state` retry window (40 attempts)
- pre-warm all involved app stores on env bootstrap (`warm_apps`)
- `open_app` now preloads the app store first (harmless no-op on this MobileGym pin)
- recycle the browser every 4 episodes (qz chromium dies after many
  screenshot-heavy episodes)
- `--disable-gpu` on chromium launch
- shim `libs/` contains ONLY chromium-specific libs (libnspr4/libnss3/libatk/
  libgbm/libdrm/libX11/libxcb/libdbus/libxkbcommon) — generic libs (libz/libzstd/
  libglib/...) are left to the qz image's system libs, matching local behaviour.

## What actually fixes alipay

The remaining failure is environmental: the qz `relay2:v2` image is a bare CUDA
image without desktop chromium libraries, and shimming them via `LD_LIBRARY_PATH`
is not sufficient for MobileGym's alipay chunk. The reliable fix is a GPU image
with the full desktop dependency set (e.g. `nvidia/cuda` + `chromium` installed
via apt), so chromium uses system libraries.

Until then, use `train_task_limit` that avoids alipay (`feedback_only_qwen4b_pilot.json`
uses 4). The 32/256/3 full run must either exclude alipay-family tasks or run on
a desktop-capable GPU image.
