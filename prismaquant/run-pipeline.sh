#!/usr/bin/env bash
# run-pipeline.sh — end-to-end PrismaQuant pipeline: probe → cost →
# allocator → native compressed-tensors export → vLLM validate.
#
# Usage:
#   MODEL_PATH=/path/to/Qwen3.6-35B-A3B \
#   WORK_DIR=./dq-runs/qwen36 \
#   FORMATS=NVFP4,FP8_DYNAMIC,BF16 \
#     (FORMATS also accepts the menu TOKEN `TESSERA`, which stands for
#      Tessera's continuous rate axis across its three serialisable families
#      -- E2M1 arity 1 and 2 on the NVFP4/W4A4 route, E4M3 arity 1 on the
#      FP8/W8A8 route. It is a token and not a format because the realisable
#      rung set depends on each unit's column count: one 0.6B Linear carries
#      thousands of legal rungs. See the TESSERA COST STAGE guard below.)
#   TARGET_BITS=4.75 \
#   VISUAL_FORMAT=BF16 \
#   CALIBRATION_MODALITY=text-only \
#   ./prismaquant/run-pipeline.sh
#
# VISUAL_FORMAT accepts any format registry name allowed by the target
# serving profile and applies to visual-encoder Linears on multimodal models.
# In text-only calibration mode it's the Phase 1 uniform override for every
# visual Linear. In multimodal mode it's the fallback applied to un-probed
# visual Linears the allocator's Fisher-driven DP didn't touch (plus the
# graceful OOM fallback on 122B-scale VLMs that can't fit the whole model in
# RAM for the multimodal pass).
#
# CALIBRATION_MODALITY (text-only | multimodal):
#   - text-only (default): body-only streaming probe + cost. Visual
#     shards emit empty pickles; allocator stamps all visual Linears
#     with --visual-format uniformly.
#   - multimodal: also runs a non-streaming second probe pass with
#     image+text calibration (synthetic stub by default; set MM_DATASET
#     to a HuggingFace dataset id to use real images). The allocator
#     treats visual Linears as regular DP candidates when real Fisher
#     stats are present. Requires ~full-model RAM; falls back to
#     text-only behavior automatically on OOM.
#
# Memory note: probe + cost peak around 90 GB on a 35B model under
# BF16 calibration. The watchdog in incremental_measure_quant_cost
# aborts cleanly on swap pressure rather than OOM-killing the host.
#
# MTP is folded into the incremental probe + cost as a built-in shard;
# mtp.* tensors are measured in the same pass as the body and land in
# the same probe/cost pickles. No separate MTP stages.
#
# LM_HEAD_FORMAT=BF16 is the historical default. A measured non-BF16 value
# (the native production rung is FP8_E4M3) is fixed outside body bpp/DP but
# inside exact artifact-byte accounting. ALLOW_PINNED=lm_head remains the
# separate research mode in which the allocator chooses the head format.

set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the source HF model directory}"
: "${WORK_DIR:?Set WORK_DIR to a writable directory for artifacts}"
: "${FORMATS:=NVFP4,FP8_DYNAMIC,BF16}"

# --- TESSERA COST STAGE -----------------------------------------------------
# The `TESSERA` menu token is priced by its OWN cost stage
# (`python -m prismaquant.tessera_campaign`), not by the incremental cost
# stages: a Tessera rung has no rate a shape-free menu can enumerate, and every
# rung costs its own encode (the "embedded ladder" is a decode-time COMPLETION
# axis and does not exist on the serialised wire), so the campaign measures a
# few anchors per (unit, family) and interpolates between them under a
# leave-one-anchor-out gate. Until that stage is wired into this orchestrator,
# a run that asks for the token must be handed the campaign's cost.pkl
# explicitly. Refusing is better than emitting a cost table with no Tessera
# column and an allocation that silently contains none.
if [[ ",${FORMATS}," == *",TESSERA,"* ]]; then
  if [[ -z "${COST_PATH_OVERRIDE:-}" ]]; then
    echo "[pipeline] ERROR: FORMATS contains the TESSERA menu token, whose cost stage is prismaquant.tessera_campaign and is not yet run by this orchestrator. Produce it first:" >&2
    echo "[pipeline]   PRISMAQUANT_TESSERA_MENU=research python -m prismaquant.tessera_campaign --model \"\$MODEL_PATH\" --out \"\$WORK_DIR/artifacts/cost.pkl\" --cache-dir \"\$WORK_DIR/artifacts/tessera-cache\"" >&2
    echo "[pipeline] then re-run with COST_PATH_OVERRIDE pointing at it and COST_MODE matching its explicit provenance.cost_mode. The available Tessera menu comes from the reviewed producer contract and the explicit serving target, not a fixed rung roster. Research mode admits serialisable rungs but does not attest them for export." >&2
    exit 2
  fi
fi
: "${TARGET_BITS:=4.75}"
: "${PARETO_TARGETS:=4.5,4.6,4.7,4.75,4.85,5.0,5.25,5.5,6.0,7.0,8.25}"
# TARGET_DISK_GB (re-vet R1, closes debt D12): the byte budget is the
# CONSTRAINT and measured KL is the OBJECTIVE. When set it OVERRIDES
# TARGET_BITS — the allocator solves exact tensor spans plus an operator-set
# non-tensor reserve, then the exporter enforces exact recursive file bytes —
# re-emits at the chosen bpp — and it flips SELECTION_MODE to
# validated-surrogate for this run (an explicit SELECTION_MODE still wins), so
# the ship pick is made on measured KL among the allocations that fit rather
# than on the surrogate knee. It also narrows the Pareto set to the ~3 rungs
# that can fit, which is what keeps the extra KL evals cheap.
: "${TARGET_DISK_GB:=}"
: "${ARTIFACT_OVERHEAD_RESERVE_BYTES:=}"
# Calibration defaults. 4x256 was the historical minimum for correctness
# validation; 32x1024 (N=32, T=1024 = 32768 tokens/sample, 32 samples)
# produces ~7% lower PPL on the resulting quantized artifact at a
# linear time cost in probe wall-time. Override for faster iteration.
: "${NSAMPLES:=32}"
: "${SEQLEN:=1024}"
# LAYERS_PER_SHARD controls how many decoder layers share one reverse
# sweep for Fisher accumulation. Larger = fewer sweeps = faster probe,
# but each shard needs more gradient + retained-activation memory.
# Default `auto` asks prismaquant.autoscale to pick from available RAM +
# model size. Set to an int (e.g. `2`) to override.
: "${LAYERS_PER_SHARD:=auto}"
# CACHE_HEADROOM_GB controls the streaming layer-cache budget
# (cache = free_RAM - headroom). Default `auto` picks from model +
# LAYERS_PER_SHARD. Set to a float to override.
: "${CACHE_HEADROOM_GB:=auto}"
: "${PREFETCH_LOOKAHEAD:=auto}"
: "${PREFETCH_WORKERS:=auto}"
: "${PREFETCH_MIN_AVAILABLE_GB:=auto}"
# GGUF lane defaults to 4x the activation rows: the GPTQ-into-k-quant
# Hessian is rank-starved at 256 rows on wide layers (rank 2.6% of a
# 9728-dim H) and degenerates toward RTN — measured on Qwen3-4B, 1024
# rows closed the render gap vs llama.cpp's imatrix quantizer from +20%
# to +7.7% KLD (top-1 at parity). See docs/lanes/gguf.md.
: "${EXPORT_CONTAINER:=compressed-tensors}"
python -m prismaquant.prismasnap_contract --model "$MODEL_PATH"
if [[ ( -e "${MODEL_PATH}/prismasnap_provenance.json" \
        || -L "${MODEL_PATH}/prismasnap_provenance.json" ) \
      && "$EXPORT_CONTAINER" != "compressed-tensors" ]]; then
  echo "[pipeline] ERROR: PrismaSnap-prepared sources are admitted only to the measured native compressed-tensors lane; EXPORT_CONTAINER=${EXPORT_CONTAINER} is unvalidated." >&2
  exit 2
fi
if [[ "$EXPORT_CONTAINER" == "nvfp4_cb" ]]; then
  # RETIRED 2026-09-02.  Rob: "put Tessera in PrismaQuant and remove Gridbook".
  # The codebook container was served only by the separately released Gridbook
  # vLLM plugin, and that lane -- its pins, its packaged runtime contract, its
  # exporter, its route-status gate and its validators -- is now at
  # archive/gridbook_lane_2026-09-02/.  A driver that still accepted this
  # container would allocate and render bytes no sanctioned runtime reads, so
  # it fails here rather than after an export's GPU hours.  The sanctioned
  # containers are compressed-tensors (vanilla vLLM), gguf, and the Tessera
  # wire.
  echo "[pipeline] ERROR: EXPORT_CONTAINER=nvfp4_cb is RETIRED. The Gridbook codebook serving lane was removed on 2026-09-02; see archive/gridbook_lane_2026-09-02/README.md. Use compressed-tensors, gguf, or the Tessera lane." >&2
  exit 2
fi
# Tessera lane knobs. TESSERA_REPO is where the pinned release's checkout
# lives: PrismaQuant NAMES Tessera's own plan translator and exporter rather
# than vendoring either (the same "named, not copied" boundary
# lane_specs/tessera.json already uses for the serve script and the route
# census), because a wire recipe with two homes is how the two halves of one
# format drift apart. TESSERA_SERVE_MODE is the plugin's single operator knob
# and is DECLARED rather than defaulted silently: it changes the footprint the
# artifact occupies and is folded into vLLM's compile-cache key.
: "${TESSERA_REPO:=/home/rob/tessera}"
# EXPORTED, not merely set: the lane preflight resolves each declared producer
# tool through the env var the DECLARATION names (lane_specs/tessera.json's
# producer_tools[].repo_env), so the child process has to be able to read it.
# It was a shell-local variable while the existence check was a loop in this
# file, which is exactly the coupling that moved.
export TESSERA_REPO
# Keep the existing unscoped default, but do not turn it into an operator's
# declaration when a v5 runtime target is requested below.
TESSERA_SERVE_MODE_EXPLICIT=${TESSERA_SERVE_MODE+x}
: "${TESSERA_SERVE_MODE:=resident}"
TESSERA_SCOPE_ARGS=()
if [[ -n "${TESSERA_PLATFORM:-}${TESSERA_RUNTIME_IMAGE:-}${TESSERA_EXECUTION_MODE:-}" ]]; then
  if [[ "$TESSERA_SERVE_MODE_EXPLICIT" != "x" ]]; then
    echo "[pipeline] ERROR: an explicit Tessera runtime target requires TESSERA_SERVE_MODE=resident|streamed; the legacy default is not a scoped declaration." >&2
    exit 2
  fi
  [[ -z "${TESSERA_PLATFORM:-}" ]] || TESSERA_SCOPE_ARGS+=(--tessera-platform "$TESSERA_PLATFORM")
  [[ -z "${TESSERA_RUNTIME_IMAGE:-}" ]] || TESSERA_SCOPE_ARGS+=(--tessera-runtime-image "$TESSERA_RUNTIME_IMAGE")
  [[ -z "${TESSERA_EXECUTION_MODE:-}" ]] || TESSERA_SCOPE_ARGS+=(--tessera-execution-mode "$TESSERA_EXECUTION_MODE")
  TESSERA_SCOPE_ARGS+=(--tessera-residency "$TESSERA_SERVE_MODE")
fi
# The inputs the allocation was PRICED under, handed back to the exporter.
# TESSERA_HESSIAN is the campaign's hessian_capture.pt (the exact XtX per unit
# the cost table and any H-aware bytes were built on); TESSERA_INPUT_SCALES is
# its input_scales.safetensors (one static input_global_scale per unit, fused
# siblings unified -- the value the W4A4 costs were scored under and the value
# the serve reads as trellis_input_global_scale). Both default to unset, and
# the lane preflight FAILS CLOSED when the allocation declares a priced
# requirement these do not satisfy (RobTand/prismaquant#193): an export
# without them would build an artifact that is not the artifact priced.
: "${TESSERA_HESSIAN:=}"
: "${TESSERA_INPUT_SCALES:=}"
# `as-allocated` plans exactly the units the allocation names and spells every
# other body Linear BF16 explicitly. `broadcast-by-role` EXTRAPOLATES a
# single-layer allocation to every depth and stamps itself as an
# extrapolation; it is never the default, because silence must not become a
# 4-bit rung.
: "${TESSERA_PLAN_COVER:=as-allocated}"
if [[ "$EXPORT_CONTAINER" == "gguf" ]]; then
  : "${ACTIVATION_ROWS_LIMIT:=1024}"
else
  : "${ACTIVATION_ROWS_LIMIT:=256}"
fi
# Calibration corpora live OUTSIDE the repo (they are built, not vendored) and
# their location is one knob. It used to default to an absolute path under a
# developer's home directory, which is unreachable for everyone else and failed
# with a HuggingFace "dataset doesn't exist on the Hub" error naming that path
# — see the fail-fast in sensitivity_probe.load_calibration. Point
# PRISMAQUANT_CALIBRATION_DIR wherever you keep them, or set DATASET directly
# to any .jsonl / .txt / HF dataset id.
PIPELINE_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${PRISMAQUANT_CALIBRATION_DIR:=${PIPELINE_SCRIPT_DIR%/}/../calibration}"
: "${DATASET:=${PRISMAQUANT_CALIBRATION_DIR%/}/diverse-v1.jsonl}"
# MINOR-M2: packed-MoE experts use a cross-domain held-out corpus for the
# GPTQ-vs-RTN do-no-harm gate (the served-validated recipe — arm E in
# moe_expert_gptq_vs_rtn — beat same-corpus/in-sample gating). Must be DISJOINT
# from DATASET. Ignored for dense models (no packed experts). Set empty to
# reproduce the historical same-corpus in-sample gate.
: "${EXPERT_GATE_DATASET:=${PRISMAQUANT_CALIBRATION_DIR%/}/xdom-gate-v1.jsonl}"
: "${DEVICE:=cuda}"
: "${EXPORT_DEVICE:=cuda}"   # CUDA ~10× faster than CPU on NVFP4 packing
# Published artifacts are packaged in ~1 GiB safetensors shards (Robert,
# 2026-08-20: "package all models using 1gb file sizes"; it was 5 GiB before).
# This mirrors export_native_compressed's own --shard-bytes default so the
# pipeline and a hand-run export agree. NOTE the scope: only the
# compressed-tensors lane shards its output. The CB lane's streaming exporter
# writes a single model.safetensors and has no shard-size concept, so it is
# NOT covered by this default.
: "${EXPORT_SHARD_BYTES:=1073741824}"
# TARGET_PROFILE is deliberately UNSET by default (re-vet R11 / debt D4). A
# hardcoded `:=vllm_packed_moe` here won `resolve_target_profile`'s explicit-
# request precedence over the architecture's own `spec.default_serving_profile`
# for every run — measured cost, 2026-07-11: 226 dense FP8 Linears silently
# coerced to BF16 on the Hy3 compressed-tensors export, because the allocator
# solved under the shell's vllm_packed_moe while export re-resolved hy_v3's
# declared `gguf`. Unset, the spec default wins; TARGET_PROFILE_DEFAULT is the
# fallback when an architecture declares nothing — never `research`, whose
# format menu is unbounded. Explicit TARGET_PROFILE still wins, so every
# in-tree launch script is bit-identical.
: "${TARGET_PROFILE:=}"
: "${TARGET_PROFILE_DEFAULT:=vllm_packed_moe}"
# ALLOW_PINNED forwards `allocator --allow-pinned`: comma-separated qname
# substrings whose profile pin is lifted so the DP places them by budget-value
# instead of force-excluding them at BF16. Empty (the default) is the historical
# behaviour exactly. This exists because the flag was reachable only by driving
# the allocator by hand, and a pinned lm_head is not a small rounding error on a
# card-sized artifact: on Qwen3.8-27B its BF16 span is 2.543 GB, i.e. 20% of a
# 13.0 GB whole-artifact budget. The allocator still enforces the flag's own
# preconditions (a cost row and probe n_params for the name) and refuses rather
# than silently allocating a name it cannot price; and profile pins that exist
# for a SERVING reason are documented as not overridable there.
: "${ALLOW_PINNED:=}"

# -----------------------------------------------------------------------
# Preflight: lane eligibility + serving-profile resolution (re-vet R6/R11).
#
# Lane eligibility is a model-profile property, not an operator preference:
# an undeclared lane does NOT fail loudly at serve time — the missing
# per-architecture loader means the runtime serves uninitialised weights and
# generates coherent-looking garbage (precedent 9a79963: Laguna, 93% of
# params). require_lane_supported refuses up front against the declared set.
# The same block resolves the serving profile ONCE so the lane gates below,
# the pipeline spec and the allocator all see the same answer.
# -----------------------------------------------------------------------
if ! TARGET_PROFILE_RESOLVED="$(
  PQ_MODEL_PATH="$MODEL_PATH" \
  PQ_EXPORT_CONTAINER="$EXPORT_CONTAINER" \
  PQ_TARGET_PROFILE="$TARGET_PROFILE" \
  PQ_TARGET_PROFILE_DEFAULT="$TARGET_PROFILE_DEFAULT" \
  python3 - <<'PY'
import os
import sys

from prismaquant.model_profiles import detect_profile
from prismaquant.serving_profiles import (
    require_lane_supported,
    resolve_target_profile,
)

profile = detect_profile(os.environ["PQ_MODEL_PATH"])
try:
    require_lane_supported(profile, os.environ["PQ_EXPORT_CONTAINER"])
except SystemExit as exc:
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from None
print(resolve_target_profile(
    profile,
    os.environ.get("PQ_TARGET_PROFILE") or None,
    default=os.environ["PQ_TARGET_PROFILE_DEFAULT"],
))
PY
)"; then
  echo "[pipeline] ERROR: preflight refused this run (export lane not declared for the architecture, or serving-profile resolution failed)." >&2
  exit 2
fi
echo "[pipeline] preflight: EXPORT_CONTAINER=${EXPORT_CONTAINER} lane OK; target profile resolves to ${TARGET_PROFILE_RESOLVED}${TARGET_PROFILE:+ (explicit TARGET_PROFILE=$TARGET_PROFILE)}"

# -----------------------------------------------------------------------
# Cost axes (re-vet R3). COST_MODE silently decided TWO independent things:
# which RENDER produces the per-(Linear, format) error, and which OBJECTIVE
# maps that error to predicted_dloss. They are now named:
#
#   COST_RENDER    ∈ {inline, cached-menu}
#       inline      — the cost stage renders each (Linear, format) itself,
#                     through the family's own qdq (weighted where the family
#                     is weighted).
#       cached-menu — the error is read off a ProductionWeightCache render of
#                     the whole format menu.
#   COST_OBJECTIVE ∈ {weight-recon, render-score, aura-adjoint}
#
# COST_MODE remains the documented spelling and keeps its exact meaning; the
# axes are the mechanism underneath, and setting them directly is equivalent:
#   local                  = inline      x weight-recon
#   production-render-score= cached-menu x render-score
#   aura                   = cached-menu x aura-adjoint
# Only those three pairs are implemented; anything else stops with the reason.
# -----------------------------------------------------------------------
if [[ -n "${COST_RENDER:-}" || -n "${COST_OBJECTIVE:-}" ]]; then
  if [[ -n "${COST_MODE:-}" ]]; then
    echo "[pipeline] ERROR: set COST_MODE or the (COST_RENDER, COST_OBJECTIVE) axes, not both — they are two spellings of one setting and a disagreement has no defensible resolution." >&2
    exit 2
  fi
  : "${COST_RENDER:=cached-menu}"
  : "${COST_OBJECTIVE:=render-score}"
  case "${COST_RENDER}|${COST_OBJECTIVE}" in
    "inline|weight-recon")           COST_MODE="local" ;;
    "cached-menu|render-score")      COST_MODE="production-render-score" ;;
    "cached-menu|aura-adjoint")      COST_MODE="aura" ;;
    "inline|aura-adjoint")
      echo "[pipeline] ERROR: COST_RENDER=inline x COST_OBJECTIVE=aura-adjoint is not implemented — the AURA adjoint consumes production-rendered dW from a format-menu cache (aura_cost --require-production-cache), which is the cached-menu render by definition." >&2
      exit 2
      ;;
    "cached-menu|weight-recon")
      echo "[pipeline] ERROR: COST_RENDER=cached-menu x COST_OBJECTIVE=weight-recon is not implemented — production_render_cost derives cost from the scores the render recorded (render-score), and the weight-recon objective is the inline stage's own measurement. Use COST_MODE=local or COST_MODE=production-render-score." >&2
      exit 2
      ;;
    *)
      echo "[pipeline] ERROR: COST_RENDER must be inline|cached-menu and COST_OBJECTIVE must be weight-recon|render-score|aura-adjoint (got '${COST_RENDER}' x '${COST_OBJECTIVE}')" >&2
      exit 2
      ;;
  esac
  echo "[pipeline] cost axes: COST_RENDER=${COST_RENDER} x COST_OBJECTIVE=${COST_OBJECTIVE} -> COST_MODE=${COST_MODE}"
fi
# COST_MODE default lives here (not in the defaults block below) because the
# lane render-faithfulness assertion runs before it.
: "${COST_MODE:=aura}"
case "$COST_MODE" in
  local)                              COST_RENDER=inline;      COST_OBJECTIVE=weight-recon ;;
  production-render-score|production-render)
                                      COST_RENDER=cached-menu; COST_OBJECTIVE=render-score ;;
  aura)                               COST_RENDER=cached-menu; COST_OBJECTIVE=aura-adjoint ;;
  *)                                  COST_RENDER=unknown;     COST_OBJECTIVE=unknown ;;
esac

# -----------------------------------------------------------------------
# Lane render-faithfulness assertion (re-vet R3), replacing the two
# `COST_MODE=local` gates.
#
# What those gates were protecting is a property of the RENDER, not of the
# objective: the render that produces the allocator's cost must be the render
# the exporter ships. The right key already existed —
# `measure_quant_cost._cost_render_uses_imatrix` decides it per FORMAT FAMILY
# (the CB families always weighted, gguf tracking PRISMAQUANT_GGUF_IMATRIX) —
# the gates just did not use it, which is why `COST_MODE=local` had to stand
# in for "weighted render" and blocked two objectives for no reason of their
# own.
#
#   COST_RENDER=inline      -> the cost stage calls the family's own qdq:
#                              faithful by construction.
#   COST_RENDER=cached-menu -> faithful iff the ProductionWeightCache render
#                              applies the same imatrix. Since CB Milestone C
#                              (R3: `col_weights` on render_production_weight)
#                              it can, given the harvested vector — so the
#                              pipeline harvests it and passes --col-weights,
#                              and this block records that requirement.
#
# NOT a promotion: AURA-on-CB is now REACHABLE, not recommended. Its −38% /
# −17.9% wins are native-lane results and CB's error surface (VQ + the expert
# route-flip floor) is a different animal; it stays opt-in pending its own
# served A/B.
# -----------------------------------------------------------------------
COST_CACHE_COL_WEIGHTS_REQUIRED=0
if [[ "$EXPORT_CONTAINER" == "gguf" ]]; then
  if ! LANE_RENDER_WEIGHTED="$(
    PQ_EXPORT_CONTAINER="$EXPORT_CONTAINER" python3 - <<'PY'
import os
import sys

from prismaquant import format_registry as fr
from prismaquant.measure_quant_cost import _cost_render_uses_imatrix

FAMILIES = {"gguf": ("gguf",)}
families = FAMILIES[os.environ["PQ_EXPORT_CONTAINER"]]
specs = [s for s in fr.list_formats() if s.family in families]
if not specs:
    print("no registered formats for families "
          f"{families}", file=sys.stderr)
    raise SystemExit(2)
weighted = {_cost_render_uses_imatrix(s) for s in specs}
if len(weighted) != 1:
    print(f"families {families} disagree on imatrix weighting: {weighted}",
          file=sys.stderr)
    raise SystemExit(2)
print("1" if weighted.pop() else "0")
PY
  )"; then
    echo "[pipeline] ERROR: could not resolve the render-faithfulness of the ${EXPORT_CONTAINER} lane's format families." >&2
    exit 2
  fi
  if [[ "$COST_RENDER" == "cached-menu" && "$LANE_RENDER_WEIGHTED" == "1" ]]; then
    COST_CACHE_COL_WEIGHTS_REQUIRED=1
    echo "[pipeline] lane ${EXPORT_CONTAINER}: COST_RENDER=cached-menu on an imatrix-weighted family -> the cost cache will be built with --col-weights (CB Milestone C). COST_OBJECTIVE=${COST_OBJECTIVE} on this lane is OPT-IN and NOT the default: its accuracy case is a native-lane result and has no served CB A/B yet."
  fi
fi

# GGUF lane consistency gates (see docs/lanes/gguf.md).
if [[ "$EXPORT_CONTAINER" == "gguf" ]]; then
  if [[ "${PRODUCTION_CACHE:-1}" != "0" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=gguf requires PRODUCTION_CACHE=0 — export_gguf requantizes the bf16 skeleton and never reads the production cache; building one burns hours rendering bytes that never ship. Set PRODUCTION_CACHE=0 PRODUCTION_RECACHE=0." >&2
    exit 2
  fi
  if [[ "$TARGET_PROFILE_RESOLVED" != "gguf" ]]; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=gguf resolves the serving profile to '${TARGET_PROFILE_RESOLVED}', not gguf (the exporter hard-fails on non-GGUF formats in the assignment). Declare default_serving_profile=gguf in the architecture spec, or set TARGET_PROFILE=gguf." >&2
    exit 2
  fi
fi

# Tessera lane consistency gates read the packaged contract and release pin,
# rather than assuming the checkpoint's class or requested runtime is served.
#
#   * the checkpoint's structure must be one the packaged contract declares
#     (a checkpoint class does not determine the structure of each unit);
#   * lane_specs/tessera.json's `served_activation_quantization.executes` must
#     EQUAL what the contract's formats[] rows imply;
#   * the pinned Tessera serving runtime must be an exact reviewed release.
#
# V5 additionally requires an explicit runtime target. The selected-unit
# check runs again immediately before translation, when the allocation exists.
# A development pin does not bypass the release export gate (tessera#17).
if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then
  if ! python3 -m prismaquant.tessera_export_lane --model "$MODEL_PATH" \
      --target-profile "$TARGET_PROFILE_RESOLVED" "${TESSERA_SCOPE_ARGS[@]}"; then
    exit 2
  fi
  # The two Tessera-repository tools this arm shells out to are NOT listed
  # here. They are declared in lane_specs/tessera.json's `producer_tools`,
  # with each tool's stability and tracking issue, and the preflight above
  # iterates that declaration (tessera_export_lane.require_producer_tools).
  # A roster in this loop was a roster in a driver: it named two paths for one
  # lane, a reader of the lane spec could not see it, and a fourth lane would
  # have needed a fourth loop.
  case "$TESSERA_SERVE_MODE" in
    resident|streamed) ;;
    *)
      echo "[pipeline] ERROR: TESSERA_SERVE_MODE must be 'resident' or 'streamed' (the two residencies the packaged contract's cells receipt); got '${TESSERA_SERVE_MODE}'." >&2
      exit 2
      ;;
  esac
  # Checked here and not only by the translator's argparse `choices`, because
  # the translator does not run until the export stage: a typo would otherwise
  # cost the whole probe/cost/render run before anything refused it, and
  # refusing before GPU hours is this block's entire purpose.
  case "$TESSERA_PLAN_COVER" in
    as-allocated|broadcast-by-role) ;;
    *)
      echo "[pipeline] ERROR: TESSERA_PLAN_COVER must be 'as-allocated' or 'broadcast-by-role'; got '${TESSERA_PLAN_COVER}'." >&2
      exit 2
      ;;
  esac
fi

# NVFP4-CB / FP8-CB codebook lane consistency gates (docs/lanes/nvfp4-cb/
# format-pipeline.md §6, LAYOUT.md). The rendering-confound half is now the
# shared render-faithfulness assertion above (R3); what remains here is the
# serving-profile and production-cache consistency the CB container needs.

if [[ "$DEVICE" != cuda* || "$EXPORT_DEVICE" != cuda* ]]; then
  echo "[pipeline] ERROR: PrismaQuant production pipeline is GPU-or-bust; DEVICE and EXPORT_DEVICE must be cuda*" >&2
  exit 2
fi
python3 - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    print("[pipeline] ERROR: CUDA is not available; refusing CPU quantization", file=sys.stderr)
    raise SystemExit(2)
PY
# Visual encoder format: fallback for visual Linears. See header docstring
# for the full text-only vs multimodal semantics. BF16 (default) is
# passthrough; non-BF16 registry formats uniformly quantize
# non-Fisher-allocated visual Linears via the existing RTN math at export time.
: "${VISUAL_FORMAT:=BF16}"
# Calibration modality. `text-only` (default) is the streaming body probe
# alone; `multimodal` adds a second non-streaming pass over the full
# model with image+text calibration so visual Linears get real Fisher
# stats. See header docstring.
: "${CALIBRATION_MODALITY:=text-only}"
# Multimodal dataset. `synthetic` is the offline stub that exercises the
# code path without network. Set to an HF dataset id (e.g.
# `HuggingFaceM4/COCO`) for real image calibration when
# CALIBRATION_MODALITY=multimodal.
: "${MM_DATASET:=synthetic}"
: "${MTP_FORMAT:=BF16}"
# Fixed output-head format. BF16 is byte-for-byte historical behavior.
# FP8_E4M3 is the native compressed-tensors production rung validated by the
# terminal-head experiment; unlike ALLOW_PINNED=lm_head it is auxiliary to the
# body DP/bpp and contributes only to the exact whole-artifact byte ledger.
: "${LM_HEAD_FORMAT:=BF16}"

# Resolve the head policy once so allocator, production-cache fill, and export
# cannot disagree about a profile pin. The Python helper also lifts every
# source/live alias of one structural head (e.g. DeepSeek's head/lm_head pair)
# while leaving unrelated profile pins intact.
if ! LM_HEAD_POLICY_TEXT="$(
  PQ_MODEL_PATH="$MODEL_PATH" \
  PQ_ALLOW_PINNED="$ALLOW_PINNED" \
  PQ_LM_HEAD_FORMAT="$LM_HEAD_FORMAT" \
  PQ_BODY_FORMATS="$FORMATS" \
  PQ_COST_PATH_OVERRIDE="${COST_PATH_OVERRIDE:-}" \
  python3 - --target-profile "${TARGET_PROFILE_RESOLVED:-}" "${TESSERA_SCOPE_ARGS[@]}" <<'PY'
import argparse
import os
import sys

from prismaquant import format_registry as fr
from prismaquant.fixed_head import (
    allow_pinned_lifts_lm_head,
    remaining_profile_pins,
)
from prismaquant.model_profiles import detect_profile
from prismaquant.serving_profiles import load_serving_profile
from prismaquant.tessera_serving_scope import (
    add_serving_scope_arguments,
    serving_target_from_args,
    unit_structure_from_profile,
)

profile = detect_profile(os.environ["PQ_MODEL_PATH"])
parser = argparse.ArgumentParser()
parser.add_argument("--target-profile", default=None)
add_serving_scope_arguments(parser)
scope_args = parser.parse_args()
try:
    serving_profile = load_serving_profile(scope_args.target_profile or None)
    target = serving_target_from_args(scope_args, target_platform=serving_profile.target_platform)
except (OSError, ValueError) as exc:
    print(f"[pipeline] ERROR: invalid Tessera serving target: {exc}", file=sys.stderr)
    raise SystemExit(2) from None
try:
    canonical = fr.get_format(os.environ["PQ_LM_HEAD_FORMAT"]).name
except Exception as exc:
    print(f"[pipeline] ERROR: invalid LM_HEAD_FORMAT: {exc}", file=sys.stderr)
    raise SystemExit(2) from None
head_context = None
if target is not None and fr.is_tessera_format_name(canonical):
    head_context = {"lm_head": target.context(unit_structure_from_profile("lm_head", profile))}
if not fr.format_is_producer_eligible(canonical, **(
        {"context_by_unit": head_context} if head_context is not None else {})):
    print(
        f"[pipeline] ERROR: LM_HEAD_FORMAT={canonical} is reader-only and "
        "cannot enter a new artifact.",
        file=sys.stderr,
    )
    raise SystemExit(2)
allow = os.environ.get("PQ_ALLOW_PINNED", "")
dp_unpinned = allow_pinned_lifts_lm_head(profile, allow)
fixed_quantized = canonical != "BF16"
if fixed_quantized and dp_unpinned:
    print(
        "[pipeline] ERROR: LM_HEAD_FORMAT fixes lm_head outside the body DP, "
        "while ALLOW_PINNED asks the DP to choose it. Drop ALLOW_PINNED for "
        "the fixed production recipe, or leave LM_HEAD_FORMAT=BF16 for the "
        "research/DP path.",
        file=sys.stderr,
    )
    raise SystemExit(2)

formats = []
seen = set()
for raw in os.environ["PQ_BODY_FORMATS"].split(","):
    value = raw.strip()
    if not value:
        continue
    if value == "TESSERA":
        # This is a token, never a shape-free FormatSpec. Its cost columns and
        # per-unit admission are resolved by the allocator. The separate cost
        # override stage-graph repair is tracked in PrismaQuant #184.
        if not os.environ.get("PQ_COST_PATH_OVERRIDE"):
            print("[pipeline] ERROR: TESSERA menu token requires COST_PATH_OVERRIDE", file=sys.stderr)
            raise SystemExit(2)
        if value not in seen:
            seen.add(value)
            formats.append(value)
        continue
    fmt = fr.get_format(value).name
    if target is not None and fr.is_tessera_format_name(fmt):
        from prismaquant.tessera_render import tessera_rung_is_serialisable

        # No unit topology exists at this name-only boundary. Check bytes can
        # be written, then let the allocator and selected-export gate ask the
        # same complete context for each actual unit; never fabricate a
        # model-wide dense or routed claim just to pass this preliminary check.
        eligible = tessera_rung_is_serialisable(fmt)
    else:
        eligible = fr.format_is_producer_eligible(fmt)
    if not eligible:
        print(
            f"[pipeline] ERROR: FORMATS contains reader-only {fmt}; legacy "
            "wire ids may be inspected but cannot enter a new cost/allocator/"
            "export menu.",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if fmt not in seen:
        seen.add(fmt)
        formats.append(value)
if fixed_quantized and canonical not in seen:
    formats.append(canonical)

print(canonical)
print("1" if fixed_quantized else "0")
print("1" if dp_unpinned else "0")
print("1" if fixed_quantized or dp_unpinned else "0")
print(",".join(formats))
if target is not None:
    print(target.platform)
for pin in remaining_profile_pins(
    profile,
    allow_pinned=allow,
    fixed_lm_head_quantized=fixed_quantized,
):
    print(pin)
PY
)"; then
  echo "[pipeline] ERROR: failed to resolve fixed/unpinned lm_head policy." >&2
  exit 2
fi
mapfile -t LM_HEAD_POLICY_LINES <<<"$LM_HEAD_POLICY_TEXT"
if (( ${#LM_HEAD_POLICY_LINES[@]} < 5 )); then
  echo "[pipeline] ERROR: lm_head policy resolver returned an incomplete result." >&2
  exit 2
fi
LM_HEAD_FORMAT_CANONICAL="${LM_HEAD_POLICY_LINES[0]}"
LM_HEAD_FIXED_QUANTIZED="${LM_HEAD_POLICY_LINES[1]}"
LM_HEAD_DP_UNPINNED="${LM_HEAD_POLICY_LINES[2]}"
LM_HEAD_RENDER_ACTIVE="${LM_HEAD_POLICY_LINES[3]}"
COST_FORMATS="${LM_HEAD_POLICY_LINES[4]}"
TESSERA_RESOLVED_PLATFORM=""
TESSERA_SCOPE_RESIDENCY=""
TESSERA_SCOPE_TARGET_PROFILE=""
if [[ ${#TESSERA_SCOPE_ARGS[@]} -gt 0 ]]; then
  if (( ${#LM_HEAD_POLICY_LINES[@]} < 6 )); then
    echo "[pipeline] ERROR: scoped policy resolver omitted the resolved Tessera platform." >&2
    exit 2
  fi
  TESSERA_RESOLVED_PLATFORM="${LM_HEAD_POLICY_LINES[5]}"
  TESSERA_SCOPE_RESIDENCY="$TESSERA_SERVE_MODE"
  TESSERA_SCOPE_TARGET_PROFILE="$TARGET_PROFILE_RESOLVED"
  REMAINING_PROFILE_PINS=("${LM_HEAD_POLICY_LINES[@]:6}")
else
  REMAINING_PROFILE_PINS=("${LM_HEAD_POLICY_LINES[@]:5}")
fi

LM_HEAD_BASE_COST_ARGS=(--no-include-lm-head)
if [[ "$LM_HEAD_RENDER_ACTIVE" == "1" ]]; then
  LM_HEAD_BASE_COST_ARGS=(--include-lm-head)
fi
LM_HEAD_AURA_ARGS=()
if [[ "$LM_HEAD_DP_UNPINNED" == "1" ]]; then
  LM_HEAD_AURA_ARGS=(--include-lm-head)
fi
PRODUCTION_CACHE_PIN_ARGS=(--skip-qnames "${REMAINING_PROFILE_PINS[@]}")
EXPORT_PIN_ARGS=(--ignore "${REMAINING_PROFILE_PINS[@]}")

# Production-cache export path. Enabled by default so export packs the same
# rendered weights that KL/polish paths measure. Re-cache is enabled by default
# after the Qwen3.5-0.8B and Qwen3-4B smoke ladder cleared vLLM eager/graph
# serving; set PRODUCTION_RECACHE=0 for an explicit no-recache ablation.
: "${PRODUCTION_CACHE:=1}"
: "${PRODUCTION_RECACHE:=1}"
: "${PRODUCTION_CACHE_MAX_ACT_ROWS:=512}"
: "${PRODUCTION_CACHE_LRU_GB:=64.0}"
: "${PRODUCTION_CACHE_PREFETCH:=require}"
# Export-side assignment prefetch (re-vet R24/D8): require on the native
# lane, so a cache that cannot serve the assignment fails loudly instead of
# silently degrading the export to an NVMe-bound crawl.
: "${EXPORT_PRODUCTION_CACHE_PREFETCH:=require}"
: "${PRODUCTION_CACHE_PREFETCH_WORKERS:=4}"
: "${PRODUCTION_RECACHE_MICROBATCH:=1}"
: "${PRODUCTION_CACHE_FORMATS:=auto}"
# assignment: render only concrete non-BF16 entries from layer_config.json.
# format-menu: render every requested format for every quantizable Linear,
# useful when intentionally building a reusable cache for reallocations.
: "${PRODUCTION_CACHE_RENDER_SCOPE:=assignment}"
: "${PRODUCTION_CACHE_LEVERS:=gptq,static_act_order,joint_scale_opt}"
: "${PRODUCTION_CACHE_DISABLE_LEVERS:=}"
# COST_MODE / COST_RENDER / COST_OBJECTIVE are resolved in the cost-axes block
# above (re-vet R3), which runs before the lane render-faithfulness assertion.
: "${PRODUCTION_RENDER_COST_NSAMPLES:=8}"
: "${PRODUCTION_RENDER_COST_SEQLEN:=1024}"
: "${PRODUCTION_RENDER_COST_SEED:=42}"
# weight_mse default since 2026-07-02 (audit M6): the legacy
# h_trace*output_mse product carries the activation energy E||x||^2 twice
# (h_trace is the weight-space Fisher trace and already contains it), a
# per-Linear multiplicative bias ~ in_features*x_rms^2. Served two-arm A/B
# at matched 4.75 bpp, 5 window draws + 32k-token PPL per model:
#   Qwen3-4B:   KL -50.8% (31/40 windows), PPL 21.82 -> 18.52 (-15.1%)
#   Qwen3-0.6B: KL -58.5% (35/40 windows), PPL 45.20 -> 34.16 (-24.4%)
# output_mse reproduces the historical objective. Target-scale (27B-class)
# confirmation is ladder debt, mirroring the damp-1.0 promotion.
: "${PRODUCTION_RENDER_COST_SCORE_FIELD:=weight_mse}"
: "${PRODUCTION_RENDER_COST_REQUIRE_SCORES:=0}"
: "${PRODUCTION_RENDER_COST_REQUIRE_OUTPUT:=1}"
: "${EXPORT_GPTQ:=auto}"
: "${EXPORT_SCALE_SWEEP:=auto}"
: "${PIPELINE_SPEC_PATH:=${WORK_DIR}/artifacts/pipeline_spec.json}"
: "${PIPELINE_SPEC_VALIDATE:=1}"
# Fisher-weighted GPTQ + Fisher output-MSE allocator are archived under
# archive/fisher_2026-05-15/ (the row-weighting + allocator-objective code
# paths remain in the live tree but are unreachable from the production
# pipeline). Any explicit override here fails fast.
: "${FISHER_WEIGHTED_GPTQ:=0}"
: "${FISHER_OUTPUT_MSE_ALLOCATOR:=0}"
for _legacy_fisher_var in FISHER_WEIGHTED_GPTQ FISHER_OUTPUT_MSE_ALLOCATOR; do
  case "${!_legacy_fisher_var}" in
    0|false|False|FALSE|no|No|NO|"") ;;
    *)
      echo "[pipeline] ERROR: $_legacy_fisher_var=${!_legacy_fisher_var} — Fisher levers are archived under archive/fisher_2026-05-15." >&2
      exit 2
      ;;
  esac
done
unset _legacy_fisher_var
case ",$PRODUCTION_CACHE_LEVERS," in
  *,fisher_gptq,*)
    echo "[pipeline] ERROR: PRODUCTION_CACHE_LEVERS includes fisher_gptq — Fisher levers are archived under archive/fisher_2026-05-15." >&2
    exit 2
    ;;
esac
export PRISMAQUANT_FISHER_OUTPUT_MSE_ALLOCATOR=0
# Selection mode. Under a byte budget the ship bpp is FIXED by the card, so
# the only open question is which of the ~3 fitting allocations measures best
# — that is validated-surrogate's job and it costs ~3 KL evals there (re-vet
# R1). Without a budget it stays opt-in: on an open-ended TARGET_BITS run it
# turns the assignment-scope render into a format-menu render (~2x) plus 11 KL
# evals to decide a bpp the card would have fixed for free. An explicit
# SELECTION_MODE always wins.
if [[ -n "$TARGET_DISK_GB" ]]; then
  : "${SELECTION_MODE:=validated-surrogate}"
else
  : "${SELECTION_MODE:=surrogate}"
fi
: "${VALIDATED_FRONTIER_NSAMPLES:=$NSAMPLES}"
: "${VALIDATED_FRONTIER_SEQLEN:=$SEQLEN}"
# Held-out selection (review criticals C3/C5): the frontier-selecting KL
# must be measured on text disjoint from probe/cost/render calibration
# (house rule: "held-out split is disjoint from cost generation"). The
# loader is prefix-stable, so skipping the first NSAMPLES windows makes
# the validation windows token-disjoint by construction. Default ON —
# the prior in-sample wiring was a bug, not a behavior. Set
# VALIDATED_FRONTIER_SKIP_CALIB=0 only to reproduce historical runs.
: "${VALIDATED_FRONTIER_SKIP_CALIB:=$NSAMPLES}"
# Optional fully separate validation corpus (overrides skip-first
# disjointness with corpus-level disjointness when provided).
: "${VALIDATED_FRONTIER_DATASET:=$DATASET}"
# The effective skip-first: corpus-level disjointness (a separate validation
# dataset) supersedes prefix-skip disjointness. Computed once so the stage
# settings hash and the CLI cannot disagree.
if [[ "$VALIDATED_FRONTIER_DATASET" == "$DATASET" ]]; then
  VALIDATED_FRONTIER_CALIB_SKIP_FIRST="$VALIDATED_FRONTIER_SKIP_CALIB"
else
  VALIDATED_FRONTIER_CALIB_SKIP_FIRST=0
fi
# M26: final selection scores full-sequence KL (the gold-metric scope, §5),
# not the last_token triage screen. Selection is still re-validated on the
# served metric before ship, but aligning the selection scope with the gold
# metric removes a screen/gold mismatch. VALIDATED_FRONTIER_KL_SCOPE=last_token
# reproduces historical selections (the flagship 27B was picked under it).
: "${VALIDATED_FRONTIER_KL_SCOPE:=full_sequence}"
# Frontier pick. `budget` = min measured KL among the rows whose exact
# exported footprint fits TARGET_DISK_GB (re-vet R1); kneedle stays the
# default without a budget and is a diagnostic on a log-linear RD curve.
if [[ -n "$TARGET_DISK_GB" ]]; then
  : "${VALIDATED_FRONTIER_PICK:=budget}"
else
  : "${VALIDATED_FRONTIER_PICK:=kneedle}"
fi
: "${VALIDATED_FRONTIER_SAT_Z:=2.0}"
# Saturation (B*) needs a real per-bpp noise floor: validate_assignments_kl's
# kl_stderr is computed across calib-repeats, so a single repeat -> stderr=0 ->
# the saturation band collapses and B* degenerates to the asymptote. Auto-bump
# repeats to 4 when PICK=saturation (CLAUDE.md's --calib-repeats>=4 discipline);
# other picks keep 1 repeat (backwards-compatible). Override explicitly to pin.
if [[ "$VALIDATED_FRONTIER_PICK" == "saturation" ]]; then
  : "${VALIDATED_FRONTIER_CALIB_REPEATS:=4}"
else
  : "${VALIDATED_FRONTIER_CALIB_REPEATS:=1}"
fi
: "${VALIDATED_SOURCE_PREFETCH:=require}"
: "${VALIDATED_SOURCE_PREFETCH_MAX_GB:=0}"
: "${VALIDATED_SOURCE_PREFETCH_HEADROOM_GB:=16}"
: "${VALIDATED_SOURCE_PREFETCH_WORKERS:=2}"
: "${VALIDATED_FRONTIER_KL_CUDA_GRAPHS:=auto}"
: "${VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE:=0}"
# hooks materializes every assignment via forward hooks in one process
# (fast, but model + full render set must co-reside: OOMs on 35B-class MoE
# in the 128 GB unified pool). inplace loops one validate_assignments_kl
# invocation per Pareto point (weights installed in place, renders paged
# per point) and merges the per-point JSONs — the run_m4_validate_inplace
# pattern that fits 35B.
: "${VALIDATED_FRONTIER_MATERIALIZATION:=hooks}"
case "$VALIDATED_FRONTIER_MATERIALIZATION" in
  hooks|inplace) ;;
  *)
    echo "[pipeline] ERROR: VALIDATED_FRONTIER_MATERIALIZATION must be hooks or inplace" >&2
    exit 2
    ;;
esac
if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
  # Header-only, pre-GPU policy gate. Hooks keeps the source model and every
  # Pareto render co-resident, so it is admitted only for a checkpoint proven
  # dense and below 35B. Unknown classifications fail closed; inplace does not
  # need a size classification because it pages one assignment per process.
  python3 -m prismaquant.pipeline \
    --check-frontier-materialization "$MODEL_PATH" \
    --frontier-materialization "$VALIDATED_FRONTIER_MATERIALIZATION" \
    || exit $?
fi
# COST_MODE=aura settings (defaults = the recipes that produced the regen-27b
# and 35B arm-E wins; see .claude/prismaquant-handover + memory notes).
: "${AURA_COST_NPROBES:=32}"
: "${AURA_COST_NSAMPLES:=8}"
: "${AURA_COST_SEQLEN:=128}"
: "${AURA_COST_CALIB_SEED:=42}"
: "${AURA_COST_LINEAR_CHUNKS:=8}"
: "${AURA_COST_PROBE_MICROBATCH:=8}"
: "${AURA_COST_MIN_FREE_GIB:=18}"
: "${AURA_COST_STREAMING:=0}"
: "${AURA_COST_CHECKPOINT_DIR:=}"
# auto: fp32 when the fp32-resident model fits with the min-free headroom
# (27B-class, the regen recipe), else bf16 (35B-class — fp32 is ~140 GiB
# against the 121 GiB pool and OOM-kills the box).
: "${AURA_COST_DTYPE:=auto}"
# Empirical packed-expert unit-KL stage (MoE hybrid): serving-unit RTN KL
# measured end-to-end, FP8 kept in the menu (real-KL rejects it, no bans).
: "${AURA_EXPERT_NSAMPLES:=16}"
: "${AURA_EXPERT_SEQLEN:=512}"
# Imatrix (column-weight) harvest + ladder interpolation. These are named CB_*
# for the lane that first needed them; the harvest is what the GGUF lane and
# every imatrix-weighted cost cache / production cache read, so it stays.
# `CB_EXPERT_EMPIRICAL` went with the Gridbook codebook lane on 2026-09-02: it
# only ever selected the [2d-CB] empirical packed-expert stage on an
# `EXPORT_CONTAINER=nvfp4_cb` run, and that container now fails closed above.
# See archive/gridbook_lane_2026-09-02/README.md.
: "${CB_EXPERT_NSAMPLES:=16}"
: "${CB_EXPERT_SEQLEN:=512}"
: "${CB_EXPERT_SAMPLE:=0}"
: "${CB_LADDER_INTERP:=0}"
: "${CB_COL_WEIGHTS:=${WORK_DIR}/artifacts/cb_col_weights.pkl}"
# Activation-fair pricing (ultraplan P5a,
# docs/audits/ultraplan_perf_2026-08-01.md §6). The allocator's cost
# precedence prices the W4A4-vs-W8A8 activation contract ONLY on the measured
# output_mse branch, and the two levers above are exactly what removes that
# branch from most rows of a production run: CB_EXPERT_SAMPLE /
# PRISMAQUANT_EXPERT_COST_SAMPLE make measure_quant_cost stamp
# output_mse_measured=False on every packed-expert row, and CB_LADDER_INTERP=1
# fills interpolated rungs weight-only. On those rows NVFP4-CB is credited
# with its cheaper index stream and charged none of its A-side cost. The
# allocator now calibrates one per-format-family correction per run from the
# measured rows that DO exist and applies it to the weight-only ones.
# 1 (default) = calibrate wherever the inputs exist; 0 = reproduce a pre-0.5.3
# artifact's pricing bit-for-bit. It is NOT a knob to tune — the run refuses
# outright rather than hand the DP a half-corrected (mixed-scale) menu.
: "${ACTIVATION_FAIR_PRICING:=1}"
export PRISMAQUANT_ACTIVATION_FAIR_PRICING="${ACTIVATION_FAIR_PRICING}"

# ---------------------------------------------------------------------------
# Hard serving constraints — the allocator's SECOND selection axis
# (ultraplan P5c; docs/lanes/nvfp4-cb/format-speed-policy.md §1)
# ---------------------------------------------------------------------------
# Policy §1's production problem is "minimize predicted quality loss SUBJECT TO
# exact bytes <= B, p95 TTFT <= SLO_prefill, p95 ITL <= SLO_decode_itl,
# p05 TPS >= SLO_decode_tps, resident + KV + peak_scratch <= device budget".
# Latency is NEVER blended into the objective: there is no lambda and no
# phase-weighted serve_ms. An assignment that misses an SLO is INFEASIBLE and
# is removed from the candidate set; the objective stays minimum predicted
# Delta-loss over the survivors, with the existing ratchet tie-break intact.
#
# ALL OF THESE ARE OFF BY DEFAULT. Leave SERVE_DISPATCH_TABLE empty and the
# allocator behaves exactly as it did before P5c, with a provenance stamp in
# selection.json recording that no serving constraint was evaluated (so an
# artifact never implies a latency claim it did not make).
#
# SERVE_DISPATCH_TABLE — measured per-(format-family, phase, M-regime, lane)
#   serving costs, every row citing its source document, date, GPU identity,
#   measured quantity, units and the derivation that produced the ratio. A row
#   without a source is refused at load. The in-tree EXAMPLE table was built
#   from published codebook-lane measurements and went to
#   archive/gridbook_lane_2026-09-02/prismaquant/serve_dispatch_tables/ with
#   that lane on 2026-09-02; it was PROPOSAL DATA, never a qualified serving
#   model for any hardware. There is no in-tree example today: supply a
#   measured table or leave this empty.
# SERVE_WORKLOAD_MIX — which M-regimes the workload actually runs, e.g.
#   "prefill:dense_prefill_1400=1.0,decode:decode_batch1=1.0". Per-phase
#   weights must sum to 1.0. There is deliberately no default: policy §1
#   forbids "a default workload mix hidden in the allocator".
# SLO_* / SERVE_DEVICE_BUDGET_BYTES — the operator's hard limits. Prefill and
#   decode are separate constraints because a format can move them in opposite
#   directions (FP8-CB measures 1.44x on dense prefill and at parity on
#   batch-1 decode).
# SERVE_KV_BYTES / SERVE_PEAK_SCRATCH_BYTES — operator inputs to the device
#   memory constraint. The allocator models resident WEIGHT bytes exactly and
#   models neither of these, so it will not guess them.
#
# Whatever these produce is PROPOSAL DATA. Per NATIVE-PARITY and policy §1,
# per-layer timing tables generate candidate assignments; only the served
# protocol promotes one.
: "${SERVE_DISPATCH_TABLE:=}"
: "${SERVE_WORKLOAD_MIX:=}"
: "${SLO_PREFILL_P95_TTFT_MS:=}"
: "${SLO_DECODE_P95_ITL_MS:=}"
: "${SLO_DECODE_P05_TPS:=}"
: "${SERVE_DEVICE_BUDGET_BYTES:=}"
: "${SERVE_KV_BYTES:=0}"
: "${SERVE_PEAK_SCRATCH_BYTES:=0}"
if [[ -z "$SERVE_DISPATCH_TABLE" ]] && [[ -n "$SLO_PREFILL_P95_TTFT_MS$SLO_DECODE_P95_ITL_MS$SLO_DECODE_P05_TPS" ]]; then
  echo "[pipeline] ERROR: a latency SLO was set but SERVE_DISPATCH_TABLE is empty. Latency constraints are priced from MEASURED rows; there is no built-in cost model to fall back on, and inventing one is exactly what the constrained-Pareto axis exists to prevent (ultraplan_perf_2026-08-01 §6, P5c). Point SERVE_DISPATCH_TABLE at a measured table for your own hardware; the retired codebook lane's example table is at archive/gridbook_lane_2026-09-02/ and was PROPOSAL DATA." >&2
  exit 2
fi

PROBE_PATH="${WORK_DIR}/artifacts/probe.pkl"
case "$COST_MODE" in
  local)
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    COST_PATH="${BASE_COST_PATH}"
    PRODUCTION_RENDER_COST_CACHE_PATH=""
    PRODUCTION_RENDER_COST_CACHE_DIR=""
    # One user knob drives BOTH ladder wirings (dense local cost + the
    # empirical expert stage): CB_LADDER_INTERP=1.
    if [[ "$CB_LADDER_INTERP" == "1" ]]; then
      export PRISMAQUANT_CB_LADDER_INTERP=1
    fi
    ;;
  grouped-kl)
    echo "[pipeline] ERROR: COST_MODE=grouped-kl — the grouped-KL (fusion-matched) cost surrogate is archived under archive/grouped_kl_2026-05-28. It fixed a local allocator non-monotonicity but LOST the shipped vLLM A/B on Qwen3.6-27B (worse exact vLLM KL and direct WikiText PPL than the shipped 5.5 artifact); see archive/grouped_kl_2026-05-28/README.md. Use production-render-score (default), aura, or local." >&2
    exit 2
    ;;
  production-render-score|production-render)
    # FAIL-CLOSED: this mode is UNLICENSED on any CB/CBL-containing menu.
    # Its score field is `weight_mse` (since audit M6), and the per-unit
    # factorization mse(e,K) ~= s_e * g(K) — the assumption the whole
    # adaptive-render/allocation story rests on — FAILS in weight currency
    # across a codebook-basis change: CV over experts of
    # weight_mse_CBL/weight_mse_lattice is monotone in rung, 0.088 (K28) ->
    # 0.224 (K48), with 8 of 10 rung-pairs breaching the 0.10 bar. The same
    # six planes pass lattice->lattice at CV 0.067/0.056, so it is not a
    # cohort artifact. Mechanism: a learned book is fit to the POOLED weight
    # distribution (redistributing error across experts rather than scaling
    # it), and CBL is itself selected under an imatrix-weighted weight metric
    # (`err = err * wq` in the CB qdq) — shaped in one currency,
    # measured in another. It was also already shown to mis-rank LDLQ.
    # Allocating a CB menu on this estimator means allocating in the currency
    # that demonstrably does not transfer. Use aura (the default) or local.
    if [[ "${FORMATS:-}" == *_CB_* ]]; then
      echo "[pipeline] ERROR: COST_MODE=$COST_MODE is unlicensed on a CB/CBL menu (FORMATS=${FORMATS:-unset}). Its score field is weight_mse, the currency in which the per-unit factorization FAILS across a codebook-basis change (CV 0.088 at K28 -> 0.224 at K48; 8/10 rung-pairs breach the 0.10 bar, while lattice->lattice passes at 0.067/0.056). Activation currency holds where weight currency does not. This spelling remains valid only for reproducing pre-CB artifacts on non-CB menus. Use COST_MODE=aura (default) or local." >&2
      exit 2
    fi
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_cache.pkl"
    PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_weight_cache"
    ;;
  production-render-staged|production-render-tail)
    echo "[pipeline] ERROR: COST_MODE=$COST_MODE — the staged (NVFP4-first, promote-the-tail) production-render cost is archived under archive/production_render_staged_2026-07-30. Its own 27B result doc is a refusal: the last-token-KL screen IMPROVED (0.0232 vs 0.0280) while direct WikiText PPL REGRESSED (10.83 vs 8.33) — \"Do not ship\" (docs/results/production_render_staged_27b_results_2026-05-21.md). Promote on the serving metric, not the screen. Use production-render-score (default), aura, or local." >&2
    exit 2
    ;;
  aura)
    # AURA downstream-KL-adjoint cost (aura_cost.py): predicted_dloss from
    # KL-Fisher probes x production-rendered dW. Served wins: -38% KL @4B,
    # -17.9% @27B vs the h_trace x output_mse baseline; regen-27b (#1
    # artifact) and the 35B arm-E hybrid both ran this recipe. On MoE
    # models the smooth cost is route-flip-blind for routed experts, so
    # packed experts get MEASURED empirical unit-KL costs instead
    # (prismaquant.expert_empirical_cost) merged into one hybrid payload.
    BASE_COST_PATH="${WORK_DIR}/artifacts/cost_baseline.pkl"
    COST_PATH="${WORK_DIR}/artifacts/cost.pkl"
    AURA_COST_RAW="${WORK_DIR}/artifacts/cost_aura.pkl"
    if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
      # Principle #8: the cost's rendered dW, the frontier's measured KL,
      # and the exported bytes all come from ONE format-menu cache — build
      # the frontier cache early (identical settings to stage [4/4], which
      # then skip-if-exists) and point aura_cost at it. This is the
      # regen-27b prodcache_menu.pkl pattern, and it halves the ~60 GB
      # double-render a separate render-score cache would cost on 35B.
      PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_weight_cache_frontier_raw.pkl"
      PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_weight_cache_frontier"
    else
      PRODUCTION_RENDER_COST_CACHE_PATH="${WORK_DIR}/artifacts/production_render_score_cache.pkl"
      PRODUCTION_RENDER_COST_CACHE_DIR="${WORK_DIR}/artifacts/production_render_score_weight_cache"
    fi
    ;;
  *)
    echo "[pipeline] ERROR: COST_MODE must be local, production-render-score, or aura" >&2
    exit 2
    ;;
esac

# An explicit prepriced input is immutable caller data, not a cache to rebuild.
# Validate before probe/GPU work and keep the allocator's original input path.
PREPRICED_COST_REPORT=""
if [[ -n "${COST_PATH_OVERRIDE:-}" ]]; then
  PREPRICED_COST_REPORT="${WORK_DIR}/artifacts/prepriced_cost_input.json"
  if ! python3 -m prismaquant.prepriced_cost \
      --path "$COST_PATH_OVERRIDE" --cost-mode "$COST_MODE" \
      --model "$MODEL_PATH" --report "$PREPRICED_COST_REPORT"; then
    exit 2
  fi
  COST_PATH="$COST_PATH_OVERRIDE"
fi

case "${HADAMARD_DUQUANT:-}" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *)
    echo "[pipeline] ERROR: Hadamard-DuQuant is archived under archive/hdq_2026-05-14 and is not available on the production path." >&2
    exit 2
    ;;
esac
case ",$PRODUCTION_CACHE_LEVERS," in
  *,hadamard_duquant,*)
    echo "[pipeline] ERROR: Hadamard-DuQuant is archived under archive/hdq_2026-05-14." >&2
    exit 2
    ;;
esac
case "${MULTI_SHOT_PASSES:-1}" in
  ""|1) ;;
  *)
    echo "[pipeline] ERROR: MULTI_SHOT_PASSES=${MULTI_SHOT_PASSES} — multi-shot recalibration is archived under archive/multi_shot_2026-05-19 (Qwen3-4B production-cal showed ΔKL=0 at 5/5 budgets; small-cal cross-eval showed mean -42.5% gap-closed with one budget regressing -154%). Not a production lever." >&2
    exit 2
    ;;
esac
case "${ALLOC_PROPAGATED_SENSITIVITY_REPORT:-}" in
  "") ;;
  *)
    echo "[pipeline] ERROR: ALLOC_PROPAGATED_SENSITIVITY_REPORT=${ALLOC_PROPAGATED_SENSITIVITY_REPORT} — L3 propagated sensitivity is archived under archive/l3_propagated_2026-07-30. The cascade it belonged to is retired: the L2 fixed point beat additive L1 by only -1.5% while AURA beat L1 by -38.5%, cross-layer residuals were +5-12% diffuse with 3 of 1180 pairs significant, and L3-polish-of-many is in the graveyard for non-additivity. One faithful cost + real-KL selection replaced it; see docs/ARCHITECTURE.md §11." >&2
    exit 2
    ;;
esac
case "${PRODUCTION_CACHE_UNION:-0}" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *)
    echo "[pipeline] ERROR: PRODUCTION_CACHE_UNION=${PRODUCTION_CACHE_UNION} — the smart-union frontier cache is archived under archive/union_cache_2026-07-30. It pre-decided which Linears deserved an FP8 rung from a percentile of the NVFP4 output_mse surrogate, denying the real-KL frontier candidates it never got to measure (principle 1). Default 0 since it landed; no shipped artifact used it. The frontier renders the full format menu." >&2
    exit 2
    ;;
esac
case "${MSE_PROMOTION:-0}" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *)
    echo "[pipeline] ERROR: MSE_PROMOTION=${MSE_PROMOTION} — post-frontier MSE promotion is archived under archive/mse_promotion_2026-07-30. It rewrote the measured-KL frontier point by a local output_mse_per_bit ranking AFTER selection; on Qwen3.6-35B (Phase 1) it removed 86% of stored local MSE and landed KL 0.0898 / PPL 9.81 — beating the strategic baseline but LOSING to both the shipped 4.75 artifact and the 5.16 kneedle. Superseded by the AURA cost, which places those bits inside the DP. No shipped run carries layer_config_before_mse_promotion.json." >&2
    exit 2
    ;;
esac

mkdir -p "${WORK_DIR}"/{artifacts,act,work,logs,exported}

case "$SELECTION_MODE" in
  surrogate|validated-surrogate) ;;
  *)
    echo "[pipeline] ERROR: SELECTION_MODE must be surrogate or validated-surrogate" >&2
    exit 2
    ;;
esac

echo "[pipeline] config:"
echo "  MODEL_PATH=$MODEL_PATH"
echo "  WORK_DIR=$WORK_DIR"
echo "  FORMATS=$FORMATS  TARGET_BITS=$TARGET_BITS"
echo "  TARGET_DISK_GB=${TARGET_DISK_GB:-<unset>}  EXPORT_CONTAINER=$EXPORT_CONTAINER"
echo "  TARGET_PROFILE=${TARGET_PROFILE:-<unset, spec-resolved>} -> $TARGET_PROFILE_RESOLVED (default $TARGET_PROFILE_DEFAULT)"
echo "  ALLOW_PINNED=${ALLOW_PINNED:-<none>}"
echo "  LM_HEAD_FORMAT=$LM_HEAD_FORMAT -> $LM_HEAD_FORMAT_CANONICAL  fixed_quantized=$LM_HEAD_FIXED_QUANTIZED dp_unpinned=$LM_HEAD_DP_UNPINNED"
echo "  COST_FORMATS=$COST_FORMATS  remaining_profile_pins=${REMAINING_PROFILE_PINS[*]:-<none>}"
echo "  NSAMPLES=$NSAMPLES SEQLEN=$SEQLEN LAYERS_PER_SHARD=$LAYERS_PER_SHARD"
echo "  PREFETCH_LOOKAHEAD=$PREFETCH_LOOKAHEAD PREFETCH_WORKERS=$PREFETCH_WORKERS"
echo "  ACTIVATION_ROWS_LIMIT=$ACTIVATION_ROWS_LIMIT"
echo "  VISUAL_FORMAT=$VISUAL_FORMAT"
echo "  MTP_FORMAT=$MTP_FORMAT"
echo "  EXPORT_SHARD_BYTES=$EXPORT_SHARD_BYTES ($((EXPORT_SHARD_BYTES / 1024 / 1024)) MiB per output shard)"
echo "  CALIBRATION_MODALITY=$CALIBRATION_MODALITY  MM_DATASET=$MM_DATASET"
echo "  PRODUCTION_CACHE=$PRODUCTION_CACHE PRODUCTION_RECACHE=$PRODUCTION_RECACHE"
echo "  PRODUCTION_CACHE_FORMATS=$PRODUCTION_CACHE_FORMATS"
echo "  PRODUCTION_CACHE_RENDER_SCOPE=$PRODUCTION_CACHE_RENDER_SCOPE"
echo "  PRODUCTION_CACHE_LEVERS=$PRODUCTION_CACHE_LEVERS"
echo "  PRODUCTION_CACHE_DISABLE_LEVERS=$PRODUCTION_CACHE_DISABLE_LEVERS"
echo "  COST_MODE=$COST_MODE"
if [[ "$COST_MODE" == "production-render-score" || "$COST_MODE" == "production-render" ]]; then
  echo "  PRODUCTION_RENDER_COST_NSAMPLES=$PRODUCTION_RENDER_COST_NSAMPLES PRODUCTION_RENDER_COST_SEQLEN=$PRODUCTION_RENDER_COST_SEQLEN PRODUCTION_RENDER_COST_SEED=$PRODUCTION_RENDER_COST_SEED SCORE_FIELD=$PRODUCTION_RENDER_COST_SCORE_FIELD"
fi
if [[ "$COST_MODE" == "aura" ]]; then
  echo "  AURA_COST_NPROBES=$AURA_COST_NPROBES AURA_COST_NSAMPLES=$AURA_COST_NSAMPLES AURA_COST_SEQLEN=$AURA_COST_SEQLEN AURA_COST_CALIB_SEED=$AURA_COST_CALIB_SEED AURA_COST_DTYPE=$AURA_COST_DTYPE STREAMING=$AURA_COST_STREAMING"
  echo "  AURA_EXPERT_NSAMPLES=$AURA_EXPERT_NSAMPLES AURA_EXPERT_SEQLEN=$AURA_EXPERT_SEQLEN VALIDATED_FRONTIER_MATERIALIZATION=$VALIDATED_FRONTIER_MATERIALIZATION"
fi
echo "  EXPORT_GPTQ=$EXPORT_GPTQ EXPORT_SCALE_SWEEP=$EXPORT_SCALE_SWEEP"
echo "  PIPELINE_SPEC_PATH=$PIPELINE_SPEC_PATH"
echo "  WALK_GATE_EXECUTION=${WALK_GATE_EXECUTION:-auto->fake} WALK_GATE_SEQLEN=${WALK_GATE_SEQLEN:-8} PRISMAQUANT_WALK_GATE_OVERRIDE=${PRISMAQUANT_WALK_GATE_OVERRIDE:+<set>}"
echo "  PRISMAQUANT_NVFP4_SCALE_RULE=${PRISMAQUANT_NVFP4_SCALE_RULE:-static_6}"
echo "  PRODUCTION_CACHE_LRU_GB=$PRODUCTION_CACHE_LRU_GB PRODUCTION_CACHE_PREFETCH=$PRODUCTION_CACHE_PREFETCH"
echo "  VALIDATED_FRONTIER_SKIP_CALIB=$VALIDATED_FRONTIER_SKIP_CALIB VALIDATED_FRONTIER_DATASET=$VALIDATED_FRONTIER_DATASET"
echo "  SELECTION_MODE=$SELECTION_MODE VALIDATED_FRONTIER_NSAMPLES=$VALIDATED_FRONTIER_NSAMPLES VALIDATED_FRONTIER_SEQLEN=$VALIDATED_FRONTIER_SEQLEN VALIDATED_FRONTIER_PICK=$VALIDATED_FRONTIER_PICK"
echo

PIPELINE_SPEC_ARGS=(
  python3 -m prismaquant.pipeline
  --write-default-production "$PIPELINE_SPEC_PATH"
  --render-mechanisms "$PRODUCTION_CACHE_LEVERS"
  --disable-render-mechanisms "$PRODUCTION_CACHE_DISABLE_LEVERS"
  --model-path "$MODEL_PATH"
  --work-dir "$WORK_DIR"
  --formats "$FORMATS"
  --target-bits "$TARGET_BITS"
  --target-profile "$TARGET_PROFILE_RESOLVED"
  --calibration-modality "$CALIBRATION_MODALITY"
  --selection-mode "$SELECTION_MODE"
  --production-cache "$PRODUCTION_CACHE"
  --production-recache "$PRODUCTION_RECACHE"
)
case "$PIPELINE_SPEC_VALIDATE" in
  0|false|False|FALSE|no|No|NO|"") ;;
  *) PIPELINE_SPEC_ARGS+=(--validate) ;;
esac
"${PIPELINE_SPEC_ARGS[@]}"


# -----------------------------------------------------------------------
# Settings-hash guard (review critical C4; re-vet R5 closes debt D6).
#
# Skip-if-exists stages must refuse to reuse artifacts built under different
# quality-affecting settings — silent reuse is the rendering-confound class
# that has invalidated A/Bs before. WHICH settings key each artifact is
# `pipeline.py`'s single real job (STAGE_SETTINGS_KEYS): the shell supplies
# every value below, pipeline.py projects them onto each artifact's declared
# key set and emits STAGE_SETTINGS_PATH; `require_stage_settings <artifact>
# <stage> [LATE=value ...]` then reads the projection instead of re-deciding
# the key set at each call site. That is what stops the eleventh stage from
# arriving with no guard, and what stops ten stages from each holding a
# different opinion of what their artifact depends on.
#
# Contract: mismatch = exit 2 naming the stale file. A MISSING manifest (or a
# stage that never recorded one) only WARNs, so pre-guard artifacts are never
# invalidated.
# -----------------------------------------------------------------------
RENDER_ENV_SETTINGS=(
  "PRISMAQUANT_NVFP4_SCALE_RULE=${PRISMAQUANT_NVFP4_SCALE_RULE:-}"
  # Manifest default must match the code default (gptq_damp_sweep_enabled:
  # OFF since 2026-06-12) so recorded provenance matches the render math.
  "PRISMAQUANT_GPTQ_DAMP_SWEEP=${PRISMAQUANT_GPTQ_DAMP_SWEEP:-0}"
  "PRISMAQUANT_GPTQ_DAMP=${PRISMAQUANT_GPTQ_DAMP:-}"
  "PRISMAQUANT_ACT_CLIP_QUANTILE=${PRISMAQUANT_ACT_CLIP_QUANTILE:-0.999}"
  "PRODUCTION_CACHE_LEVERS=$PRODUCTION_CACHE_LEVERS"
  "PRODUCTION_CACHE_DISABLE_LEVERS=${PRODUCTION_CACHE_DISABLE_LEVERS:-}"
)

STAGE_SETTINGS_PATH="${WORK_DIR}/artifacts/stage_settings.json"
STAGE_SETTINGS_ENV=(
  "MODEL_PATH=$MODEL_PATH"
  "DATASET=$DATASET"
  "NSAMPLES=$NSAMPLES"
  "SEQLEN=$SEQLEN"
  "FORMATS=$FORMATS"
  "COST_FORMATS=$COST_FORMATS"
  "TARGET_BITS=$TARGET_BITS"
  "CALIBRATION_MODALITY=$CALIBRATION_MODALITY"
  "ALLOW_PINNED=$ALLOW_PINNED"
  "LM_HEAD_FORMAT=$LM_HEAD_FORMAT_CANONICAL"
  "LM_HEAD_RENDER_ACTIVE=$LM_HEAD_RENDER_ACTIVE"
  "LM_HEAD_DP_UNPINNED=$LM_HEAD_DP_UNPINNED"
  "ACTIVATION_ROWS_LIMIT=$ACTIVATION_ROWS_LIMIT"
  "COST_MODE=$COST_MODE"
  "SELECTION_MODE=$SELECTION_MODE"
  "PRODUCTION_CACHE_RENDER_SCOPE=$PRODUCTION_CACHE_RENDER_SCOPE"
  "PRODUCTION_RENDER_COST_NSAMPLES=$PRODUCTION_RENDER_COST_NSAMPLES"
  "PRODUCTION_RENDER_COST_SEQLEN=$PRODUCTION_RENDER_COST_SEQLEN"
  "PRODUCTION_RENDER_COST_SEED=$PRODUCTION_RENDER_COST_SEED"
  "PRODUCTION_RENDER_COST_SCORE_FIELD=$PRODUCTION_RENDER_COST_SCORE_FIELD"
  "PRODUCTION_RENDER_COST_REQUIRE_SCORES=$PRODUCTION_RENDER_COST_REQUIRE_SCORES"
  "PRODUCTION_RENDER_COST_REQUIRE_OUTPUT=$PRODUCTION_RENDER_COST_REQUIRE_OUTPUT"
  "AURA_COST_NPROBES=$AURA_COST_NPROBES"
  "AURA_COST_NSAMPLES=$AURA_COST_NSAMPLES"
  "AURA_COST_SEQLEN=$AURA_COST_SEQLEN"
  "AURA_COST_CALIB_SEED=$AURA_COST_CALIB_SEED"
  "AURA_COST_DTYPE=$AURA_COST_DTYPE"
  "AURA_COST_STREAMING=$AURA_COST_STREAMING"
  "AURA_COST_CHECKPOINT_DIR=$AURA_COST_CHECKPOINT_DIR"
  "AURA_EXPERT_NSAMPLES=$AURA_EXPERT_NSAMPLES"
  "AURA_EXPERT_SEQLEN=$AURA_EXPERT_SEQLEN"
  "CB_EXPERT_NSAMPLES=$CB_EXPERT_NSAMPLES"
  "CB_EXPERT_SEQLEN=$CB_EXPERT_SEQLEN"
  "CB_EXPERT_SAMPLE=$CB_EXPERT_SAMPLE"
  "CB_LADDER_INTERP=$CB_LADDER_INTERP"
  "ACTIVATION_FAIR_PRICING=$ACTIVATION_FAIR_PRICING"
  "SERVE_DISPATCH_TABLE=${SERVE_DISPATCH_TABLE:-}"
  "SERVE_WORKLOAD_MIX=${SERVE_WORKLOAD_MIX:-}"
  "SLO_PREFILL_P95_TTFT_MS=${SLO_PREFILL_P95_TTFT_MS:-}"
  "SLO_DECODE_P95_ITL_MS=${SLO_DECODE_P95_ITL_MS:-}"
  "SLO_DECODE_P05_TPS=${SLO_DECODE_P05_TPS:-}"
  "SERVE_DEVICE_BUDGET_BYTES=${SERVE_DEVICE_BUDGET_BYTES:-}"
  "SERVE_KV_BYTES=${SERVE_KV_BYTES:-0}"
  "SERVE_PEAK_SCRATCH_BYTES=${SERVE_PEAK_SCRATCH_BYTES:-0}"
  # CB_SCALE_CODING lost its shell DEFAULT on 2026-09-02 with the Gridbook
  # lane (archive/gridbook_lane_2026-09-02/), but it keeps its settings-hash
  # entry, exactly like the CB keys around it: the CB render plumbing still
  # reads it (nvfp4_cb_footprint.py, debt D34), so a persisted cost/render
  # artifact must still invalidate when an operator sets it.
  "CB_SCALE_CODING=${CB_SCALE_CODING:-}"
  "CB_CODEBOOK_SOURCE=${CB_CODEBOOK_SOURCE:-}"
  "CB_CODEBOOK_SOURCE_SCOPE=${CB_CODEBOOK_SOURCE_SCOPE:-}"
  "CB_CODEBOOK_BUNDLE=${CB_CODEBOOK_BUNDLE:-}"
  "CB_LEARNED_TRAINER_VERSION=${CB_LEARNED_TRAINER_VERSION:-}"
  "CB_LEARNED_PROMOTION_RECEIPT=${CB_LEARNED_PROMOTION_RECEIPT:-}"
  "CB_LEARNED_PROMOTION_RECEIPT_SHA256="
  "CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE=${CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE:-}"
  "CB_LEARNED_SOURCE_MODEL_IDENTITY_SHA256="
  "CB_ROUTED_MOE_BOOK_SELECTION=${CB_ROUTED_MOE_BOOK_SELECTION:-}"
  "CB_ROUTED_MOE_BOOK_SELECTION_SHA256=${CB_ROUTED_MOE_BOOK_SELECTION_SHA256:-}"
  "CB_ROUTED_BOOK_KEYING=${CB_ROUTED_BOOK_KEYING:-}"
  "CB_ALLOW_PER_ROLE_BOOKS=${CB_ALLOW_PER_ROLE_BOOKS:-}"
  "CB_SCALE_SWEEP=${CB_SCALE_SWEEP:-}"
  "CB_SCALE_SWEEP_SCOPE=${CB_SCALE_SWEEP_SCOPE:-}"
  "CB_ACTIVATION_SCOPE=${CB_ACTIVATION_SCOPE:-}"
  "CB_IMATRIX_SOURCE=${CB_IMATRIX_SOURCE:-activation-cache}"
  "PRISMAQUANT_CB_LDLQ=${PRISMAQUANT_CB_LDLQ:-}"
  "PRISMAQUANT_CB_MINCHAIN=${PRISMAQUANT_CB_MINCHAIN:-}"
  "PRISMAQUANT_CB_MINCHAIN_ANCHORS=${PRISMAQUANT_CB_MINCHAIN_ANCHORS:-}"
  "PRISMAQUANT_CB_MINCHAIN_HOLDBACKS=${PRISMAQUANT_CB_MINCHAIN_HOLDBACKS:-}"
  "PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED=${PRISMAQUANT_CB_MINCHAIN_AUDIT_SEED:-}"
  "PRISMAQUANT_CB_MINCHAIN_BACKSTOP=${PRISMAQUANT_CB_MINCHAIN_BACKSTOP:-}"
  "PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN=${PRISMAQUANT_CB_MINCHAIN_AUDIT_MEDIAN:-}"
  "PRISMAQUANT_CB_MINCHAIN_AUDIT_P95=${PRISMAQUANT_CB_MINCHAIN_AUDIT_P95:-}"
  "PRISMAQUANT_CB_ENCODE_TIER=${PRISMAQUANT_CB_ENCODE_TIER:-}"
  "VALIDATED_FRONTIER_DATASET=$VALIDATED_FRONTIER_DATASET"
  "VALIDATED_FRONTIER_NSAMPLES=$VALIDATED_FRONTIER_NSAMPLES"
  "VALIDATED_FRONTIER_SEQLEN=$VALIDATED_FRONTIER_SEQLEN"
  "VALIDATED_FRONTIER_CALIB_REPEATS=$VALIDATED_FRONTIER_CALIB_REPEATS"
  "VALIDATED_FRONTIER_CALIB_SKIP_FIRST=$VALIDATED_FRONTIER_CALIB_SKIP_FIRST"
  "VALIDATED_FRONTIER_KL_SCOPE=$VALIDATED_FRONTIER_KL_SCOPE"
  "TESSERA_PLAN_COVER=$TESSERA_PLAN_COVER"
  "TESSERA_PLATFORM=$TESSERA_RESOLVED_PLATFORM"
  "TESSERA_RUNTIME_IMAGE=${TESSERA_RUNTIME_IMAGE:-}"
  "TESSERA_EXECUTION_MODE=${TESSERA_EXECUTION_MODE:-}"
  "TESSERA_RESIDENCY=$TESSERA_SCOPE_RESIDENCY"
  "TESSERA_TARGET_PROFILE=$TESSERA_SCOPE_TARGET_PROFILE"
  "${RENDER_ENV_SETTINGS[@]}"
)
STAGE_SETTINGS_ARGS=()
for _kv in "${STAGE_SETTINGS_ENV[@]}"; do
  STAGE_SETTINGS_ARGS+=(--setting "$_kv")
done
unset _kv
python3 -m prismaquant.pipeline \
  --write-stage-settings "$STAGE_SETTINGS_PATH" \
  "${STAGE_SETTINGS_ARGS[@]}"

# -----------------------------------------------------------------------
# Cost-table provenance (re-vet R2 precondition (i)).
#
# `cost.pkl` is the SAME path under every COST_MODE (local /
# production-render-score / aura), so a bare `[[ -f ]]` reuse test silently
# allocates on the previous mode's estimator while the log says otherwise —
# D6's silent-reuse class landing exactly on a mode change. Every producer now
# stamps provenance["cost_mode"]; reuse is conditional on it matching. This is
# the in-tree pattern the CB hybrid already used for its own merge probe.
# Unstamped tables (pre-R2) warn and are reused, never invalidated.
# -----------------------------------------------------------------------
cost_table_cost_mode() {
  python3 - "$1" <<'COSTPROV'
import pickle
import sys

try:
    with open(sys.argv[1], "rb") as fh:
        payload = pickle.load(fh)
    prov = payload.get("provenance", {}) if isinstance(payload, dict) else {}
    print(str((prov or {}).get("cost_mode", "") or ""))
except Exception:
    print("")
COSTPROV
}

cost_table_reusable() {
  local path="$1" stamped
  [[ -f "$path" ]] || return 1
  stamped="$(cost_table_cost_mode "$path")"
  if [[ -z "$stamped" ]]; then
    echo "[pipeline] WARNING: $path carries no provenance['cost_mode'] (predates the R2 stamp); reusing it under COST_MODE=${COST_MODE} unverified"
    return 0
  fi
  if [[ "$stamped" != "$COST_MODE" ]]; then
    echo "[pipeline] cost table $path was produced under COST_MODE=${stamped} but this run is COST_MODE=${COST_MODE} -> REBUILDING (reusing it would allocate on the other estimator; re-vet R2)"
    return 1
  fi
  return 0
}

require_stage_settings() {
  local artifact="$1" stage="$2"; shift 2
  local overrides=()
  local kv
  for kv in "$@"; do
    overrides+=(--setting "$kv")
  done
  python3 -m prismaquant.pipeline --check-stage-settings \
    --stage-settings "$STAGE_SETTINGS_PATH" \
    --artifact "$artifact" --stage "$stage" \
    "${overrides[@]+"${overrides[@]}"}"
  local rc=$?
  if [[ $rc -ne 0 ]]; then exit "$rc"; fi
}

# -----------------------------------------------------------------------
# CB / GGUF col-weights (imatrix) harvest — ONE definition, four call sites
# (pre-cost local expert costs, the cached-menu cost cache, [2d-CB], and the
# exporter). The vector is the exporter's own importance weighting, including
# the synthesized per-expert gate_up/down_proj entries a raw activation
# harvest cannot contain — which is why every weighted render takes it as an
# argument rather than deriving E[x^2] locally. Skip-if-exists; override
# CB_COL_WEIGHTS to supply a pre-built {qname: (in_features,)} pickle.
# -----------------------------------------------------------------------
harvest_cb_col_weights() {
  local stage="$1"
  local imatrix_source="${CB_IMATRIX_SOURCE:-activation-cache}"
  require_stage_settings "$CB_COL_WEIGHTS" cb-col-weights \
    "CB_IMATRIX_SOURCE=$imatrix_source"
  if [[ -f "$CB_COL_WEIGHTS" ]]; then
    echo "[pipeline] ${stage} CB col-weights exist, skipping"
    return 0
  fi
  echo "[pipeline] ${stage} harvesting CB col-weights (imatrix source=${imatrix_source}) ..."
  CB_ACT_DIR="${WORK_DIR}/act" CB_COL_WEIGHTS="$CB_COL_WEIGHTS" \
  CB_PROBE_PATH="$PROBE_PATH" CB_IMATRIX_SOURCE="$imatrix_source" \
  MODEL_PATH="$MODEL_PATH" CB_STAGE="$stage" python3 - <<'PY'
import json, os, pickle
from pathlib import Path

from prismaquant.cb_imatrix import (
    canonical_imatrix_sha256,
    imatrix_from_probe_file,
)
from prismaquant.export_gguf import build_imatrix_from_act_cache
from prismaquant.moe_imatrix import synthesize_packed_expert_col_weights
act_dir = os.environ["CB_ACT_DIR"]
out = os.environ["CB_COL_WEIGHTS"]
stage = os.environ["CB_STAGE"]
source = os.environ["CB_IMATRIX_SOURCE"]
if source == "probe":
    cw, provenance = imatrix_from_probe_file(os.environ["CB_PROBE_PATH"])
    if not provenance.get("calibration_hash"):
        raise SystemExit(
            "[pipeline] ERROR: probe-derived CB imatrix has no calibration hash"
        )
elif source == "activation-cache":
    cw = build_imatrix_from_act_cache(act_dir)
    provenance = {
        "schema": "prismaquant.cb_imatrix.activation_cache_rows.v1",
        "source": "activation-cache",
    }
else:
    raise SystemExit(
        "[pipeline] ERROR: CB_IMATRIX_SOURCE must be probe or activation-cache"
    )
if not cw:
    raise SystemExit(
        f"[pipeline] ERROR: no imatrix values from {source!r}; the "
        f"weighted render needs a col-weights (imatrix) vector per target. "
        f"Run the probe+cost stages first (they populate {act_dir}).")
added = synthesize_packed_expert_col_weights(
    os.environ["MODEL_PATH"], act_dir, cw)
if added:
    print(f"[pipeline] {stage} synthesized {len(added)} packed-expert "
          f"imatrix entries (gate_up pool / down_proj routed replay)")
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "wb") as fh:
    pickle.dump(cw, fh)
provenance = {
    **provenance,
    "source": source,
    "final_entries": len(cw),
    "synthesized_packed_entries": sorted(added),
    "final_value_sha256": canonical_imatrix_sha256(cw),
}
Path(out + ".provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(
    f"[pipeline] {stage} wrote {out}: {len(cw)} entries "
    f"value_sha256={provenance['final_value_sha256']}"
)
PY
}

ensure_cb_learned_bundle() {
  if [[ "${CB_CODEBOOK_SOURCE_SCOPE:-none}" == "none" ]]; then
    return 0
  fi
  harvest_cb_col_weights "$1"
  local col_sha
  col_sha="$(sha256sum "$CB_COL_WEIGHTS" | cut -d' ' -f1)"
  local trainer_version="${CB_LEARNED_TRAINER_VERSION:-v1}"
  local promotion_sha=""
  if [[ -n "${CB_LEARNED_PROMOTION_RECEIPT:-}" ]]; then
    promotion_sha="$(sha256sum "$CB_LEARNED_PROMOTION_RECEIPT" | cut -d' ' -f1)"
  fi
  local source_identity_sha=""
  if [[ -n "${CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE:-}" ]]; then
    source_identity_sha="$(sha256sum "$CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE" | cut -d' ' -f1)"
  fi
  # The keying is part of a book's identity, so a bundle built under the other
  # rule must rebuild loudly rather than be reused.
  require_stage_settings "$CB_CODEBOOK_BUNDLE" cb-learned-bundle \
    "CB_COL_WEIGHTS_SHA256=$col_sha" \
    "CB_ROUTED_BOOK_KEYING=${CB_ROUTED_BOOK_KEYING:-stack}" \
    "CB_LEARNED_TRAINER_VERSION=$trainer_version" \
    "CB_LEARNED_PROMOTION_RECEIPT_SHA256=$promotion_sha" \
    "CB_LEARNED_SOURCE_MODEL_IDENTITY_SHA256=$source_identity_sha"
  if [[ -f "$CB_CODEBOOK_BUNDLE" ]]; then
    # Full name/shape/digest/cell validation; a same-path replacement never
    # counts as an immutable bundle merely because the file exists.
    python3 - "$CB_CODEBOOK_BUNDLE" <<'PY'
import sys
from prismaquant.cb_learned_bundle import load_bundle
bundle = load_bundle(sys.argv[1])
print(f"[pipeline] learned CB bundle verified: {bundle.path} "
      f"sha256={bundle.bundle_content_sha256}")
PY
    return 0
  fi
  echo "[pipeline] $1 training immutable learned CB cells before any cost/cache/KL render ..."
  local routed_book_args=()
  if [[ -n "${CB_ROUTED_MOE_BOOK_SELECTION:-}" ]]; then
    routed_book_args+=(
      --routed-moe-book-selection "$CB_ROUTED_MOE_BOOK_SELECTION"
    )
  fi
  local imatrix_args=(--col-weights "$CB_COL_WEIGHTS")
  local learned_v2_args=()
  if [[ "$trainer_version" == "v2" ]]; then
    imatrix_args=(--imatrix-probe "$PROBE_PATH")
    learned_v2_args=(
      --promotion-receipt "$CB_LEARNED_PROMOTION_RECEIPT"
      --source-model-identity-cache "$CB_LEARNED_SOURCE_MODEL_IDENTITY_CACHE"
    )
  fi
  python3 -m prismaquant.build_cb_learned_bundle \
    --model-dir "$MODEL_PATH" \
    "${imatrix_args[@]}" \
    --formats "$FORMATS" \
    --output "$CB_CODEBOOK_BUNDLE" \
    --device "$DEVICE" \
    --trainer-version "$trainer_version" \
    --routed-book-keying "${CB_ROUTED_BOOK_KEYING:-stack}" \
    "${learned_v2_args[@]+"${learned_v2_args[@]}"}" \
    "${routed_book_args[@]}"
}

formats_contain_cb() {
  python3 - "$1" <<'PY'
import sys
from prismaquant import format_registry as fr
from prismaquant.nvfp4_cb_footprint import is_cb_format

formats = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
raise SystemExit(0 if any(is_cb_format(fr.get_format(item).name) for item in formats) else 1)
PY
}

# -----------------------------------------------------------------------
# 1. Sensitivity probe (per-Linear empirical Fisher diagonal trace,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
require_stage_settings "${PROBE_PATH}" probe
if [[ ! -f "${PROBE_PATH}" ]]; then
  echo "[pipeline] [1/4] running sensitivity probe ..."
  python3 -m prismaquant.incremental_probe \
    --model "$MODEL_PATH" \
    --dataset "$DATASET" \
    --nsamples "$NSAMPLES" --seqlen "$SEQLEN" \
    --device "$DEVICE" --dtype bf16 \
    --output "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --work-dir "${WORK_DIR}/work" \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --prefetch-lookahead "$PREFETCH_LOOKAHEAD" \
    --prefetch-workers "$PREFETCH_WORKERS" \
    --prefetch-min-available-gb "$PREFETCH_MIN_AVAILABLE_GB" \
    --activation-rows-limit "$ACTIVATION_ROWS_LIMIT" \
    --calibration-modality "$CALIBRATION_MODALITY" \
    --mm-dataset "$MM_DATASET" \
    --mm-nsamples 8 --mm-max-text-len 128 \
    2>&1 | tee "${WORK_DIR}/logs/probe.log"
else
  # Reuse guard: make sure the pre-existing probe.pkl matches the
  # currently-requested calibration modality. Silently reusing a
  # text-only probe under multimodal (or vice versa) would produce
  # an assignment calibrated for the wrong activation distribution
  # — visible later as bad PPL that's hard to root-cause. Fail loud
  # and point the user at the file to delete.
  probe_modality=$(python3 -c "
import pickle, sys
try:
    with open(sys.argv[1], 'rb') as f:
        blob = pickle.load(f)
    meta = blob.get('meta', {}) if isinstance(blob, dict) else {}
    m = meta.get('calibration_modality') or meta.get('modality') or 'text-only'
    print(m)
except Exception as e:
    print(f'__error__:{e}', file=sys.stderr)
    sys.exit(2)
" "${PROBE_PATH}" 2>/dev/null || echo "__unknown__")
  if [[ "${probe_modality}" == "__unknown__" ]]; then
    echo "[pipeline] [1/4] probe.pkl exists but its calibration_modality"
    echo "             could not be read. Aborting to avoid mixing probes."
    echo "             Delete it explicitly to regenerate:"
    echo "               rm ${PROBE_PATH}"
    exit 2
  fi
  if [[ "${probe_modality}" != "${CALIBRATION_MODALITY}" ]]; then
    echo "[pipeline] [1/4] ABORT: probe.pkl was calibrated for"
    echo "             modality='${probe_modality}' but this run requests"
    echo "             CALIBRATION_MODALITY='${CALIBRATION_MODALITY}'."
    echo "             Reusing the probe would silently produce an"
    echo "             assignment calibrated on the wrong activations."
    echo ""
    echo "             Delete the stale probe to regenerate:"
    echo "               rm ${PROBE_PATH}"
    echo "             Or unset CALIBRATION_MODALITY to match the probe."
    exit 2
  fi
  echo "[pipeline] [1/4] probe.pkl exists (modality=${probe_modality}), skipping"
fi

# -----------------------------------------------------------------------
# 2. Cost measurement (per-(Linear, format) measured RTN error,
#    body + MTP in one pass)
# -----------------------------------------------------------------------
if [[ -n "$PREPRICED_COST_REPORT" ]]; then
  echo "[pipeline] [2/4] using validated prepriced input: $COST_PATH (cost builders skipped)"
else
require_stage_settings "${BASE_COST_PATH}" base-cost
# Under COST_MODE=local the baseline IS the allocator's cost table, so it is
# also gated on the stamped cost_mode. Under the other modes cost_baseline.pkl
# is mode-agnostic (the same measured RTN error feeds every estimator) and is
# reused across mode changes on purpose.
BASE_COST_REUSABLE=1
if [[ ! -f "${BASE_COST_PATH}" ]]; then
  BASE_COST_REUSABLE=0
elif [[ "$BASE_COST_PATH" == "$COST_PATH" ]] && ! cost_table_reusable "$BASE_COST_PATH"; then
  BASE_COST_REUSABLE=0
fi
if [[ "$BASE_COST_REUSABLE" == "0" ]]; then
  echo "[pipeline] [2/4] measuring per-(layer, format) cost ..."
  python3 -m prismaquant.incremental_measure_quant_cost \
    --model "$MODEL_PATH" \
    --cost-mode "$COST_MODE" \
    --probe "${PROBE_PATH}" \
    --activation-cache-dir "${WORK_DIR}/act" \
    --formats "$COST_FORMATS" \
    --output "${BASE_COST_PATH}" \
    --work-dir "${WORK_DIR}/work" \
    --device "$DEVICE" --dtype bf16 \
    --mode batched --chunk-size 256 \
    --layers-per-shard "$LAYERS_PER_SHARD" \
    --skip-missing-activations \
    "${LM_HEAD_BASE_COST_ARGS[@]}" \
    --swap-grow-limit-mb "${SWAP_GROW_LIMIT_MB:-2048}" \
    2>&1 | tee "${WORK_DIR}/logs/cost.log"
else
  echo "[pipeline] [2/4] baseline cost exists, skipping"
fi
# Fisher output-MSE allocator cost-status check removed; the archive guard
# at the top of this file errors out before reaching here if the var is set.

# R3 / CB Milestone C: on a weighted-render lane the cached-menu cost cache
# must render with the exporter's imatrix or it is not the bytes that ship.
# Harvested once and shared with [2d-CB] and the exporter.
COST_CACHE_COL_WEIGHT_ARGS=()
if [[ "$COST_CACHE_COL_WEIGHTS_REQUIRED" == "1" ]]; then
  harvest_cb_col_weights "[2b/4] cost-cache"
  COST_CACHE_COL_WEIGHT_ARGS=(--col-weights "$CB_COL_WEIGHTS")
fi

if [[ "$COST_MODE" == "production-render-score" || "$COST_MODE" == "production-render" ]]; then
  require_stage_settings "$PRODUCTION_RENDER_COST_CACHE_PATH" render-cost-cache
  if [[ ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2b/4] rendering production weights for allocator cost ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$COST_FORMATS" \
      --render-scope format-menu \
      --activation-cache-dir "${WORK_DIR}/act" \
      "${PRODUCTION_CACHE_PIN_ARGS[@]}" \
      "${COST_CACHE_COL_WEIGHT_ARGS[@]+"${COST_CACHE_COL_WEIGHT_ARGS[@]}"}" \
      --n-calib-samples "$PRODUCTION_RENDER_COST_NSAMPLES" \
      --calib-seqlen "$PRODUCTION_RENDER_COST_SEQLEN" \
      --calib-seed "$PRODUCTION_RENDER_COST_SEED" \
      --dataset "$DATASET" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_score_cache.log"
  else
    echo "[pipeline] [2b/4] production-render cost cache exists, skipping"
  fi
  require_stage_settings "$COST_PATH" render-cost
  if ! cost_table_reusable "$COST_PATH"; then
    echo "[pipeline] [2c/4] synthesizing production-render allocator cost ..."
    PROD_RENDER_COST_ARGS=()
    case "$PRODUCTION_RENDER_COST_REQUIRE_SCORES" in
      1|true|True|TRUE|yes|Yes|YES|on|On|ON)
        PROD_RENDER_COST_ARGS+=(--require-render-scores)
        ;;
    esac
    case "$PRODUCTION_RENDER_COST_REQUIRE_OUTPUT" in
      0|false|False|FALSE|no|No|NO|off|Off|OFF|"") ;;
      *)
        PROD_RENDER_COST_ARGS+=(--require-output-metric)
        ;;
    esac
    python3 -m prismaquant.production_render_cost \
      --cost-mode "$COST_MODE" \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --baseline-cost "$BASE_COST_PATH" \
      --output "$COST_PATH" \
      --formats "$COST_FORMATS" \
      --score-field "$PRODUCTION_RENDER_COST_SCORE_FIELD" \
      "${PROD_RENDER_COST_ARGS[@]}" \
      2>&1 | tee "${WORK_DIR}/logs/production_render_cost.log"
  else
    echo "[pipeline] [2c/4] production-render allocator cost exists, skipping"
  fi
fi

if [[ "$COST_MODE" == "aura" ]]; then
  AURA_EXECUTION_ARGS=(
    --hook-harvest
    --gradient-checkpointing
    --n-linear-chunks "$AURA_COST_LINEAR_CHUNKS"
    --probe-microbatch "$AURA_COST_PROBE_MICROBATCH"
  )
  AURA_EXPERT_EXECUTION_ARGS=()
  case "$AURA_COST_STREAMING" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON)
      AURA_COST_STREAMING=1
      if [[ "$AURA_COST_CHECKPOINT_DIR" != /* ]]; then
        echo "[pipeline] ERROR: AURA_COST_STREAMING=1 requires an absolute AURA_COST_CHECKPOINT_DIR" >&2
        exit 2
      fi
      AURA_EXECUTION_ARGS=(
        --streaming
        --checkpoint-dir "$AURA_COST_CHECKPOINT_DIR"
        --resume
      )
      # The two checkpoint schemas both own manifest.json, so the empirical
      # routed-expert tail gets a deterministic child directory rather than
      # colliding with the smooth AURA adjoint's root. expert_empirical_cost
      # binds this resume to its own source/config/calibration identity.
      AURA_EXPERT_EXECUTION_ARGS=(
        --streaming
        --checkpoint-dir "${AURA_COST_CHECKPOINT_DIR%/}/expert-empirical-cost"
        --resume
      )
      ;;
    0|false|False|FALSE|no|No|NO|off|Off|OFF)
      AURA_COST_STREAMING=0
      if [[ -n "$AURA_COST_CHECKPOINT_DIR" ]]; then
        echo "[pipeline] ERROR: AURA_COST_CHECKPOINT_DIR requires AURA_COST_STREAMING=1" >&2
        exit 2
      fi
      ;;
    *)
      echo "[pipeline] ERROR: AURA_COST_STREAMING must be 0 or 1" >&2
      exit 2
      ;;
  esac
  # [2b] Production-faithful dW cache for the AURA adjoint. Under
  # SELECTION_MODE=validated-surrogate this IS the frontier cache (identical
  # path + settings to stage [4/4], which then skip-if-exists): ONE
  # format-menu cache supplies the cost's rendered dW, the frontier's
  # measured bytes, and the exported bytes (principle #8) — the regen-27b
  # prodcache_menu.pkl pattern.
  AURA_CACHE_FORMATS="$(python3 - "$COST_FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon != "BF16" and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
  if [[ -z "$AURA_CACHE_FORMATS" ]]; then
    echo "[pipeline] ERROR: COST_MODE=aura has no non-BF16 formats in FORMATS" >&2
    exit 2
  fi
  require_stage_settings "$PRODUCTION_RENDER_COST_CACHE_PATH" aura-dw-cache \
    "AURA_CACHE_FORMATS=$AURA_CACHE_FORMATS"
  if [[ ! -f "$PRODUCTION_RENDER_COST_CACHE_PATH" ]]; then
    echo "[pipeline] [2b/4] building format-menu production cache for AURA dW ..."
    python3 -m prismaquant.build_production_cache \
      --model "$MODEL_PATH" \
      --output "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --formats "$AURA_CACHE_FORMATS" \
      --activation-cache-dir "${WORK_DIR}/act" \
      "${PRODUCTION_CACHE_PIN_ARGS[@]}" \
      --dataset "$DATASET" \
      --n-calib-samples "$NSAMPLES" \
      --calib-seqlen "$SEQLEN" \
      --dtype bf16 \
      --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
      --enable "$PRODUCTION_CACHE_LEVERS" \
      --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
      --cache-dir "$PRODUCTION_RENDER_COST_CACHE_DIR" \
      --render-scope format-menu \
      "${COST_CACHE_COL_WEIGHT_ARGS[@]+"${COST_CACHE_COL_WEIGHT_ARGS[@]}"}" \
      $(if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then echo "--render-packed-experts"; fi) \
      2>&1 | tee "${WORK_DIR}/logs/aura_dw_cache.log"
  else
    echo "[pipeline] [2b/4] AURA dW production cache exists, skipping"
  fi

  # [2c] The AURA cost itself: KL-Fisher probes x production-rendered dW.
  # Profile-declared routed experts are deliberately omitted here (the smooth adjoint is
  # route-flip-blind on them) and costed empirically in [2d].
  require_stage_settings "$AURA_COST_RAW" aura-cost
  if [[ ! -f "$AURA_COST_RAW" ]]; then
    echo "[pipeline] [2c/4] measuring AURA downstream-KL-adjoint cost ..."
    python3 -m prismaquant.aura_cost \
      --model "$MODEL_PATH" \
      --cost-mode "$COST_MODE" \
      --output "$AURA_COST_RAW" \
      --formats "$COST_FORMATS" \
      --production-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
      --require-production-cache \
      --n-probes "$AURA_COST_NPROBES" \
      --n-calib-samples "$AURA_COST_NSAMPLES" \
      --calib-seqlen "$AURA_COST_SEQLEN" \
      --calib-seed "$AURA_COST_CALIB_SEED" \
      --dtype "$AURA_COST_DTYPE" \
      --dataset "$DATASET" \
      "${AURA_EXECUTION_ARGS[@]}" \
      --min-free-gib "$AURA_COST_MIN_FREE_GIB" \
      --accurate-chunk-bytes \
      --allow-packed-expert-omission \
      "${LM_HEAD_AURA_ARGS[@]+"${LM_HEAD_AURA_ARGS[@]}"}" \
      2>&1 | tee "${WORK_DIR}/logs/aura_cost.log"
  else
    echo "[pipeline] [2c/4] AURA cost exists, skipping"
  fi

  # [2d] Hybrid finalize: measured empirical unit-KL costs for any omitted
  # routed experts (packed or per-expert Linear; FP8 kept in the menu —
  # real-KL rejects it, no bans),
  # plus sidecar (MTP/visual) row backfill from the baseline cost. Backfilled
  # rows carry the baseline estimator and are recorded in provenance.
  require_stage_settings "$COST_PATH" aura-hybrid-cost
  if ! cost_table_reusable "$COST_PATH"; then
    OMITTED_EXPERTS="$(python3 - "$AURA_COST_RAW" <<'PY'
import pickle
import sys

payload = pickle.load(open(sys.argv[1], "rb"))
print(len(payload.get("provenance", {}).get("omitted_packed_experts", []) or []))
PY
)"
    if [[ "$OMITTED_EXPERTS" != "0" ]]; then
      echo "[pipeline] [2d/4] measuring empirical routed-expert unit-KL costs (${OMITTED_EXPERTS} omitted targets; hybrid merge) ..."
      python3 -m prismaquant.expert_empirical_cost \
        --model "$MODEL_PATH" \
        --cost-mode "$COST_MODE" \
        --output "$COST_PATH" \
        --formats "$FORMATS" \
        --dataset "$DATASET" \
        --n-calib-samples "$AURA_EXPERT_NSAMPLES" \
        --calib-seqlen "$AURA_EXPERT_SEQLEN" \
        --merge-base "$AURA_COST_RAW" \
        --backfill-base "$BASE_COST_PATH" \
        "${COST_CACHE_COL_WEIGHT_ARGS[@]+"${COST_CACHE_COL_WEIGHT_ARGS[@]}"}" \
        "${AURA_EXPERT_EXECUTION_ARGS[@]+"${AURA_EXPERT_EXECUTION_ARGS[@]}"}" \
        2>&1 | tee "${WORK_DIR}/logs/expert_empirical_cost.log"
    else
      echo "[pipeline] [2d/4] no routed experts omitted; finalizing AURA cost (sidecar backfill) ..."
      COST_MODE="$COST_MODE" python3 - "$AURA_COST_RAW" "$BASE_COST_PATH" "$COST_PATH" <<'PY'
import os
import pickle
import sys

from prismaquant.expert_empirical_cost import backfill_missing_from_base

payload = pickle.load(open(sys.argv[1], "rb"))
payload.setdefault("stats", {})
payload.setdefault("costs", {})
base = pickle.load(open(sys.argv[2], "rb"))
added = backfill_missing_from_base(payload, base)
prov = dict(payload.get("provenance", {}) or {})
if added:
    prov["backfilled_from_base"] = added
    prov["backfill_base"] = sys.argv[2]
# re-vet R2 precondition (i): stamp the mode that produced this table.
prov["cost_mode"] = os.environ.get("COST_MODE", "")
payload["provenance"] = prov
with open(sys.argv[3], "wb") as fh:
    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
print(f"[pipeline] aura cost finalized -> {sys.argv[3]} "
      f"(backfilled {len(added)} sidecar rows: {added})")
PY
    fi
  else
    echo "[pipeline] [2d/4] AURA allocator cost exists, skipping"
  fi
fi
fi  # no prepriced input: normal cost measurement/finalization


# -----------------------------------------------------------------------
# 3. Allocator (multi-choice knapsack over per-layer formats)
# -----------------------------------------------------------------------
if [[ -n "$TARGET_DISK_GB" ]]; then
  echo "[pipeline] [3/4] running allocator under a ${TARGET_DISK_GB}GB byte budget (overrides TARGET_BITS=${TARGET_BITS}) ..."
else
  echo "[pipeline] [3/4] running allocator at target=${TARGET_BITS} bpp ..."
fi
# Choose visual-sensitivity mode from calibration modality:
#   text-only → uniform (Phase 1 --visual-format path, as before)
#   multimodal → fisher (Phase 2: DP places visual Linears from real
#                        multimodal Fisher; --visual-format acts as a
#                        fallback for un-probed visual Linears only)
if [[ "$CALIBRATION_MODALITY" == "multimodal" ]]; then
  VISUAL_SENSITIVITY=fisher
else
  VISUAL_SENSITIVITY=uniform
fi
ALLOCATOR_PARETO_ARGS=()
if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
  ALLOCATOR_PARETO_DIR="${WORK_DIR}/artifacts/pareto_assignments"
  ALLOCATOR_PARETO_ARGS=(--pareto-output-dir "$ALLOCATOR_PARETO_DIR")
else
  ALLOCATOR_PARETO_DIR=""
fi
# --target-profile is passed ONLY when the operator asked for one, so the
# architecture's own spec.default_serving_profile can win (re-vet R11);
# --target-profile-default keeps the historical fallback for architectures
# that declare nothing, instead of resolve_target_profile's `research`.
ALLOCATOR_PROFILE_ARGS=(--target-profile-default "$TARGET_PROFILE_DEFAULT")
if [[ -n "$TARGET_PROFILE" ]]; then
  ALLOCATOR_PROFILE_ARGS+=(--target-profile "$TARGET_PROFILE")
fi
ALLOCATOR_PROFILE_ARGS+=("${TESSERA_SCOPE_ARGS[@]}")
if [[ -n "$ALLOW_PINNED" ]]; then
  ALLOCATOR_PROFILE_ARGS+=(--allow-pinned "$ALLOW_PINNED")
fi
# Byte budget: the constraint is the card, the objective is measured KL
# (re-vet R1). Selection reserves non-tensor bytes; export enforces the exact
# recursive regular-file ceiling.
ALLOCATOR_BUDGET_ARGS=()
if [[ -n "$TARGET_DISK_GB" ]]; then
  if [[ -z "$ARTIFACT_OVERHEAD_RESERVE_BYTES" ]]; then
    echo "[pipeline] ERROR: TARGET_DISK_GB requires ARTIFACT_OVERHEAD_RESERVE_BYTES (safetensors headers + JSON/tokenizer/processor/other output files)" >&2
    exit 2
  fi
  ALLOCATOR_BUDGET_ARGS=(
    --target-disk-gb "$TARGET_DISK_GB"
    --artifact-overhead-reserve-bytes "$ARTIFACT_OVERHEAD_RESERVE_BYTES"
  )
fi
# Hard serving constraints (ultraplan P5c). Empty SERVE_DISPATCH_TABLE ->
# no flags -> the pre-P5c allocator, byte for byte. Latency never enters the
# objective; these are constraints only.
ALLOCATOR_SERVE_ARGS=()
if [[ -n "$SERVE_DISPATCH_TABLE" ]]; then
  ALLOCATOR_SERVE_ARGS+=(--serve-dispatch-table "$SERVE_DISPATCH_TABLE")
  [[ -n "$SERVE_WORKLOAD_MIX" ]] && ALLOCATOR_SERVE_ARGS+=(--serve-workload-mix "$SERVE_WORKLOAD_MIX")
  [[ -n "$SLO_PREFILL_P95_TTFT_MS" ]] && ALLOCATOR_SERVE_ARGS+=(--slo-prefill-p95-ttft-ms "$SLO_PREFILL_P95_TTFT_MS")
  [[ -n "$SLO_DECODE_P95_ITL_MS" ]] && ALLOCATOR_SERVE_ARGS+=(--slo-decode-p95-itl-ms "$SLO_DECODE_P95_ITL_MS")
  [[ -n "$SLO_DECODE_P05_TPS" ]] && ALLOCATOR_SERVE_ARGS+=(--slo-decode-p05-tps "$SLO_DECODE_P05_TPS")
  if [[ -n "$SERVE_DEVICE_BUDGET_BYTES" ]]; then
    ALLOCATOR_SERVE_ARGS+=(
      --serve-device-budget-bytes "$SERVE_DEVICE_BUDGET_BYTES"
      --serve-kv-bytes "$SERVE_KV_BYTES"
      --serve-peak-scratch-bytes "$SERVE_PEAK_SCRATCH_BYTES"
    )
  fi
  echo "[pipeline] serving constraints ACTIVE: table=$SERVE_DISPATCH_TABLE mix='${SERVE_WORKLOAD_MIX}' (PROPOSAL DATA; the served NATIVE-PARITY protocol is the release gate)"
fi
ALLOCATOR_CB_ARGS=()
# Recheck the exact original input (including symlink target) after probe work
# and immediately before the allocator consumes it. Never rewrite that input.
if [[ -n "$PREPRICED_COST_REPORT" ]]; then
  if ! python3 -m prismaquant.prepriced_cost --verify-report "$PREPRICED_COST_REPORT"; then
    exit 2
  fi
fi
python3 -m prismaquant.allocator \
  --probe "${PROBE_PATH}" \
  --costs "${COST_PATH}" \
  --target-bits "$TARGET_BITS" \
  --formats "$FORMATS" \
  "${ALLOCATOR_PROFILE_ARGS[@]}" \
  "${ALLOCATOR_BUDGET_ARGS[@]+"${ALLOCATOR_BUDGET_ARGS[@]}"}" \
  "${ALLOCATOR_SERVE_ARGS[@]+"${ALLOCATOR_SERVE_ARGS[@]}"}" \
  "${ALLOCATOR_CB_ARGS[@]+"${ALLOCATOR_CB_ARGS[@]}"}" \
  --pareto-targets "$PARETO_TARGETS" \
  --visual-format "$VISUAL_FORMAT" \
  --visual-sensitivity "$VISUAL_SENSITIVITY" \
  --lm-head-format "$LM_HEAD_FORMAT_CANONICAL" \
  --mtp-format "$MTP_FORMAT" \
  --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
  --pareto-csv "${WORK_DIR}/artifacts/pareto.csv" \
  "${ALLOCATOR_PARETO_ARGS[@]}" \
  2>&1 | tee "${WORK_DIR}/logs/allocator.log"

# -----------------------------------------------------------------------
# 4. Production cache + native compressed-tensors export
# -----------------------------------------------------------------------
PRODUCTION_CACHE_PATH=""
if [[ "$SELECTION_MODE" == "validated-surrogate" ]] && [[ "$PRODUCTION_CACHE" == "0" || "$PRODUCTION_CACHE" == "false" || "$PRODUCTION_CACHE" == "False" ]]; then
  echo "[pipeline] ERROR: SELECTION_MODE=validated-surrogate requires PRODUCTION_CACHE=1 so KL validates production-rendered weights." >&2
  exit 2
fi
if [[ "$PRODUCTION_CACHE" != "0" && "$PRODUCTION_CACHE" != "false" && "$PRODUCTION_CACHE" != "False" ]]; then
  PROD_CACHE_DIR="${WORK_DIR}/artifacts/production_weight_cache"
  PROD_CACHE_RAW="${WORK_DIR}/artifacts/production_weight_cache_raw.pkl"
  PROD_CACHE_RECACHED="${WORK_DIR}/artifacts/production_weight_cache_recached.pkl"
  CACHE_FORMATS="$PRODUCTION_CACHE_FORMATS"
  PRODUCTION_CACHE_CB_ARGS=()
  PRODUCTION_CACHE_CB_SETTINGS=()
  if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
    if [[ -z "$ALLOCATOR_PARETO_DIR" || ! -f "$ALLOCATOR_PARETO_DIR/manifest.json" ]]; then
      echo "[pipeline] ERROR: validated-surrogate selection requires allocator pareto assignments at $ALLOCATOR_PARETO_DIR" >&2
      exit 2
    fi
    if [[ "$CACHE_FORMATS" == "auto" ]]; then
      CACHE_FORMATS="$(python3 - "$COST_FORMATS" <<'PY'
import sys
from prismaquant import format_registry as fr

seen = []
for raw in sys.argv[1].split(","):
    name = raw.strip()
    if not name:
        continue
    canon = fr.canonical_format_name(name)
    if canon != "BF16" and canon not in seen:
        seen.append(canon)
print(",".join(seen))
PY
)"
      echo "[pipeline] frontier production cache formats selected from FORMATS: ${CACHE_FORMATS:-none}"
    fi
    if [[ -z "$CACHE_FORMATS" ]]; then
      echo "[pipeline] ERROR: validated-surrogate selection has no non-BF16 cache formats" >&2
      exit 2
    fi
    if formats_contain_cb "$CACHE_FORMATS"; then
      harvest_cb_col_weights "[4/4] validated-frontier cache"
      CB_COL_WEIGHTS_SHA256=$(sha256sum "$CB_COL_WEIGHTS" | cut -d' ' -f1)
      PRODUCTION_CACHE_CB_ARGS=(--col-weights "$CB_COL_WEIGHTS")
      PRODUCTION_CACHE_CB_SETTINGS=(
        "CB_COL_WEIGHTS_SHA256=$CB_COL_WEIGHTS_SHA256"
      )
    fi
    PROD_CACHE_DIR="${PROD_CACHE_DIR}_frontier"
    PROD_CACHE_RAW="${WORK_DIR}/artifacts/production_weight_cache_frontier_raw.pkl"
    require_stage_settings "$PROD_CACHE_RAW" frontier-cache \
      "CACHE_FORMATS=$CACHE_FORMATS" \
      "${PRODUCTION_CACHE_CB_SETTINGS[@]+"${PRODUCTION_CACHE_CB_SETTINGS[@]}"}"
    if [[ ! -f "$PROD_CACHE_RAW" ]]; then
      echo "[pipeline] [4/4] building format-menu production cache for validated frontier ..."
      python3 -m prismaquant.build_production_cache \
        --model "$MODEL_PATH" \
        --output "$PROD_CACHE_RAW" \
        --formats "$CACHE_FORMATS" \
        --activation-cache-dir "${WORK_DIR}/act" \
        "${PRODUCTION_CACHE_PIN_ARGS[@]}" \
        --dataset "$DATASET" \
        --n-calib-samples "$NSAMPLES" \
        --calib-seqlen "$SEQLEN" \
        --dtype bf16 \
        --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
        --enable "$PRODUCTION_CACHE_LEVERS" \
        --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
        --cache-dir "$PROD_CACHE_DIR" \
        --render-scope format-menu \
        --render-packed-experts \
        "${PRODUCTION_CACHE_CB_ARGS[@]+"${PRODUCTION_CACHE_CB_ARGS[@]}"}" \
        2>&1 | tee "${WORK_DIR}/logs/production_cache_frontier.log"
    else
      echo "[pipeline] [4/4] frontier production cache exists, skipping"
    fi

    VALIDATED_ASSIGNMENT_ARGS=()
    while IFS=$'\t' read -r label path; do
      [[ -n "$label" && -n "$path" ]] || continue
      VALIDATED_ASSIGNMENT_ARGS+=(--assignment "${label}=${path}")
    done < <(python3 - "$ALLOCATOR_PARETO_DIR/manifest.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1]))
for row in payload.get("candidates", []):
    print(f"{row['label']}\t{row['path']}")
PY
)
    if [[ "${#VALIDATED_ASSIGNMENT_ARGS[@]}" -eq 0 ]]; then
      echo "[pipeline] ERROR: no Pareto assignments found for validated frontier" >&2
      exit 2
    fi

    VALIDATION_JSON="${WORK_DIR}/artifacts/validated_frontier_kl.json"
    VALIDATED_ASSIGNMENT_COUNT=$(( ${#VALIDATED_ASSIGNMENT_ARGS[@]} / 2 ))
    VAK_COMMON_ARGS=(
      --model "$MODEL_PATH"
      --probe "$PROBE_PATH"
      --costs "$COST_PATH"
      --base-assignment "${WORK_DIR}/artifacts/layer_config.json"
      --formats "$FORMATS"
      --dataset "$VALIDATED_FRONTIER_DATASET"
      --calib-skip-first "$VALIDATED_FRONTIER_CALIB_SKIP_FIRST"
      --n-calib-samples "$VALIDATED_FRONTIER_NSAMPLES"
      --calib-seqlen "$VALIDATED_FRONTIER_SEQLEN"
      --calib-repeats "$VALIDATED_FRONTIER_CALIB_REPEATS"
      --work-dir "${WORK_DIR}/work/validate_kl"
      --dtype bf16
      --device "$DEVICE"
      --kl-scope "$VALIDATED_FRONTIER_KL_SCOPE"
      --kl-cuda-graphs "$VALIDATED_FRONTIER_KL_CUDA_GRAPHS"
      --source-prefetch "$VALIDATED_SOURCE_PREFETCH"
      --source-prefetch-max-gb "$VALIDATED_SOURCE_PREFETCH_MAX_GB"
      --source-prefetch-headroom-gb "$VALIDATED_SOURCE_PREFETCH_HEADROOM_GB"
      --source-prefetch-workers "$VALIDATED_SOURCE_PREFETCH_WORKERS"
      --production-weight-cache "$PROD_CACHE_RAW"
      --production-cache-dir-override "$PROD_CACHE_DIR"
      --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB"
      --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH"
      --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS"
    )
    VAK_COMMON_ARGS+=(
      "${PRODUCTION_CACHE_CB_ARGS[@]+"${PRODUCTION_CACHE_CB_ARGS[@]}"}"
    )
    if [[ "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "0" && "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "false" && "$VALIDATED_DISABLE_FROZEN_WEIGHT_CACHE" != "False" ]]; then
      VAK_COMMON_ARGS+=(--disable-frozen-weight-cache)
    fi
    if [[ "$VALIDATED_FRONTIER_MATERIALIZATION" == "inplace" ]]; then
      # inplace requires exactly one assignment per process (weights are
      # installed destructively); loop the Pareto points and merge the
      # per-point JSONs. This is the memory-fit path for 35B-class MoE —
      # hooks mode needs model + all renders co-resident and OOMs the
      # 128 GB unified pool.
      echo "[pipeline] [4/4] measuring real KL for ${VALIDATED_ASSIGNMENT_COUNT} Pareto assignments (inplace, one process per point) ..."
      VAK_PART_DIR="${WORK_DIR}/artifacts/validated_frontier_kl_parts"
      mkdir -p "$VAK_PART_DIR"
      VAK_PART_FILES=()
      for ((vi = 0; vi < ${#VALIDATED_ASSIGNMENT_ARGS[@]}; vi += 2)); do
        VAK_SPEC="${VALIDATED_ASSIGNMENT_ARGS[vi + 1]}"
        VAK_LABEL="${VAK_SPEC%%=*}"
        VAK_LABEL_SAFE="${VAK_LABEL//[^A-Za-z0-9._-]/_}"
        VAK_PART="${VAK_PART_DIR}/vak_${VAK_LABEL_SAFE}.json"
        VAK_PART_FILES+=("$VAK_PART")
        require_stage_settings "$VAK_PART" frontier-kl-point
        if [[ -f "$VAK_PART" ]]; then
          echo "[pipeline] [4/4] ${VAK_LABEL}: per-point KL exists, skipping"
          continue
        fi
        python3 -m prismaquant.validate_assignments_kl \
          "${VAK_COMMON_ARGS[@]}" \
          --assignment "$VAK_SPEC" \
          --assignment-materialization inplace \
          --output "$VAK_PART" \
          2>&1 | tee "${WORK_DIR}/logs/validated_frontier_kl_${VAK_LABEL_SAFE}.log"
      done
      python3 - "$VALIDATION_JSON" "${VAK_PART_FILES[@]}" <<'PY'
import json
import sys
from pathlib import Path

out_path, *parts = sys.argv[1:]
merged = None
for part in parts:
    payload = json.loads(Path(part).read_text())
    if merged is None:
        merged = payload
    else:
        merged["results"].extend(payload.get("results", []))
if merged is None:
    raise SystemExit("[pipeline] ERROR: no per-point validation JSONs to merge")
merged["assignment_materialization"] = "inplace"
Path(out_path).write_text(json.dumps(merged, indent=2) + "\n")
print(f"[pipeline] merged {len(parts)} per-point KL JSONs -> {out_path} "
      f"({len(merged['results'])} results)")
PY
    else
      echo "[pipeline] [4/4] measuring real KL for ${VALIDATED_ASSIGNMENT_COUNT} Pareto assignments ..."
      python3 -m prismaquant.validate_assignments_kl \
        "${VAK_COMMON_ARGS[@]}" \
        "${VALIDATED_ASSIGNMENT_ARGS[@]}" \
        --assignment-materialization "$VALIDATED_FRONTIER_MATERIALIZATION" \
        --output "$VALIDATION_JSON" \
        2>&1 | tee "${WORK_DIR}/logs/validated_frontier_kl.log"
    fi

    echo "[pipeline] [4/4] selecting measured frontier point ..."
    FRONTIER_SELECT_ARGS=()
    if [[ -n "$TARGET_DISK_GB" ]]; then
      FRONTIER_SELECT_ARGS+=(--target-disk-gb "$TARGET_DISK_GB")
    fi
    python3 -m prismaquant.select_validated_frontier \
      --validation-json "$VALIDATION_JSON" \
      --mode "$VALIDATED_FRONTIER_PICK" \
      --sat-z "$VALIDATED_FRONTIER_SAT_Z" \
      "${FRONTIER_SELECT_ARGS[@]+"${FRONTIER_SELECT_ARGS[@]}"}" \
      --output-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
      --output-assignment "${WORK_DIR}/artifacts/layer_config_validated_assignment.json" \
      --output-summary "${WORK_DIR}/artifacts/validated_frontier_selection.json" \
      2>&1 | tee "${WORK_DIR}/logs/validated_frontier_select.log"

    if [[ "$PRODUCTION_RECACHE" != "0" && "$PRODUCTION_RECACHE" != "false" && "$PRODUCTION_RECACHE" != "False" ]]; then
      SELECTED_DIGEST="$(python3 - "${WORK_DIR}/artifacts/layer_config.json" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest()[:12])
PY
)"
      PROD_CACHE_RECACHED="${WORK_DIR}/artifacts/production_weight_cache_frontier_${SELECTED_DIGEST}_recached.pkl"
      require_stage_settings "$PROD_CACHE_RECACHED" frontier-recache
      if [[ ! -f "$PROD_CACHE_RECACHED" ]]; then
        echo "[pipeline] [4/4] re-fitting activation scales for selected measured-${VALIDATED_FRONTIER_PICK} assignment ..."
        python3 -m prismaquant.production_recache \
          --model "$MODEL_PATH" \
          --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --production-weight-cache "$PROD_CACHE_RAW" \
          --output "$PROD_CACHE_RECACHED" \
          --cache-dir-override "$PROD_CACHE_DIR" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --device "$DEVICE" \
          --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
          --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
          --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
          --microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          2>&1 | tee "${WORK_DIR}/logs/production_recache.log"
      else
        echo "[pipeline] [4/4] selected-assignment recached production cache exists, skipping"
      fi
      PRODUCTION_CACHE_PATH="$PROD_CACHE_RECACHED"
    else
      PRODUCTION_CACHE_PATH="$PROD_CACHE_RAW"
    fi
  elif [[ "$CACHE_FORMATS" == "auto" ]]; then
    CACHE_FORMATS="$(python3 - "${WORK_DIR}/artifacts/layer_config.json" <<'PY'
import sys
from prismaquant.production_recache import _load_assignment

assignment = _load_assignment(sys.argv[1])
formats = sorted({fmt for fmt in assignment.values() if fmt.upper() != "BF16"})
print(",".join(formats))
PY
)"
    echo "[pipeline] production cache formats selected from assignment: ${CACHE_FORMATS:-none}"
  fi
  if [[ "$SELECTION_MODE" != "validated-surrogate" \
     && -n "$CACHE_FORMATS" ]] \
     && formats_contain_cb "$CACHE_FORMATS"; then
    harvest_cb_col_weights "[4/4] production cache"
    CB_COL_WEIGHTS_SHA256=$(sha256sum "$CB_COL_WEIGHTS" | cut -d' ' -f1)
    PRODUCTION_CACHE_CB_ARGS=(--col-weights "$CB_COL_WEIGHTS")
    PRODUCTION_CACHE_CB_SETTINGS=(
      "CB_COL_WEIGHTS_SHA256=$CB_COL_WEIGHTS_SHA256"
    )
  fi
  if [[ "$SELECTION_MODE" == "validated-surrogate" ]]; then
    :
  elif [[ -z "$CACHE_FORMATS" ]]; then
    echo "[pipeline] production cache requested but no non-BF16 formats are in FORMATS; skipping cache"
  elif [[ "$PRODUCTION_RECACHE" != "0" && "$PRODUCTION_RECACHE" != "false" && "$PRODUCTION_RECACHE" != "False" ]]; then
    LC_DIGEST=$(sha256sum "${WORK_DIR}/artifacts/layer_config.json" | cut -c1-16)
    require_stage_settings "$PROD_CACHE_RECACHED" production-cache-recached \
      "ASSIGNMENT_DIGEST=$LC_DIGEST" \
      "${PRODUCTION_CACHE_CB_SETTINGS[@]+"${PRODUCTION_CACHE_CB_SETTINGS[@]}"}"
    if [[ ! -f "$PROD_CACHE_RECACHED" ]]; then
      if [[ ! -f "$PROD_CACHE_RAW" ]]; then
        echo "[pipeline] [4/4] building production cache + re-fitting activation scales ..."
        python3 -m prismaquant.build_production_cache \
          --model "$MODEL_PATH" \
          --output "$PROD_CACHE_RECACHED" \
          --formats "$CACHE_FORMATS" \
          --activation-cache-dir "${WORK_DIR}/act" \
          "${PRODUCTION_CACHE_PIN_ARGS[@]}" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
          --enable "$PRODUCTION_CACHE_LEVERS" \
          --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
          --cache-dir "$PROD_CACHE_DIR" \
          --render-scope "$PRODUCTION_CACHE_RENDER_SCOPE" \
          --render-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --recache-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --recache-microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          "${PRODUCTION_CACHE_CB_ARGS[@]+"${PRODUCTION_CACHE_CB_ARGS[@]}"}" \
          ${EXPERT_GATE_DATASET:+--expert-gate-dataset "$EXPERT_GATE_DATASET"} \
          2>&1 | tee "${WORK_DIR}/logs/production_cache.log"
      else
        echo "[pipeline] [4/4] re-fitting production activation scales ..."
        python3 -m prismaquant.production_recache \
          --model "$MODEL_PATH" \
          --layer-config "${WORK_DIR}/artifacts/layer_config.json" \
          --production-weight-cache "$PROD_CACHE_RAW" \
          --output "$PROD_CACHE_RECACHED" \
          --cache-dir-override "$PROD_CACHE_DIR" \
          --dataset "$DATASET" \
          --n-calib-samples "$NSAMPLES" \
          --calib-seqlen "$SEQLEN" \
          --dtype bf16 \
          --device "$DEVICE" \
          --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB" \
          --production-cache-prefetch "$PRODUCTION_CACHE_PREFETCH" \
          --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS" \
          --microbatch-size "$PRODUCTION_RECACHE_MICROBATCH" \
          2>&1 | tee "${WORK_DIR}/logs/production_recache.log"
      fi
    else
      echo "[pipeline] [4/4] recached production cache exists, skipping"
    fi
    PRODUCTION_CACHE_PATH="$PROD_CACHE_RECACHED"
  else
    RAW_LC_DIGEST=$(sha256sum "${WORK_DIR}/artifacts/layer_config.json" | cut -c1-16)
    require_stage_settings "$PROD_CACHE_RAW" production-cache-raw \
      "CACHE_FORMATS=$CACHE_FORMATS" "ASSIGNMENT_DIGEST=$RAW_LC_DIGEST" \
      "${PRODUCTION_CACHE_CB_SETTINGS[@]+"${PRODUCTION_CACHE_CB_SETTINGS[@]}"}"
    if [[ ! -f "$PROD_CACHE_RAW" ]]; then
      echo "[pipeline] [4/4] building production cache ..."
      python3 -m prismaquant.build_production_cache \
        --model "$MODEL_PATH" \
        --output "$PROD_CACHE_RAW" \
        --formats "$CACHE_FORMATS" \
        --activation-cache-dir "${WORK_DIR}/act" \
        "${PRODUCTION_CACHE_PIN_ARGS[@]}" \
        --dataset "$DATASET" \
        --n-calib-samples "$NSAMPLES" \
        --calib-seqlen "$SEQLEN" \
        --dtype bf16 \
        --max-act-rows "$PRODUCTION_CACHE_MAX_ACT_ROWS" \
        --enable "$PRODUCTION_CACHE_LEVERS" \
        --disable "$PRODUCTION_CACHE_DISABLE_LEVERS" \
        --cache-dir "$PROD_CACHE_DIR" \
        --render-scope "$PRODUCTION_CACHE_RENDER_SCOPE" \
        --render-layer-config "${WORK_DIR}/artifacts/layer_config.json" \
        "${PRODUCTION_CACHE_CB_ARGS[@]+"${PRODUCTION_CACHE_CB_ARGS[@]}"}" \
        ${EXPERT_GATE_DATASET:+--expert-gate-dataset "$EXPERT_GATE_DATASET"} \
        2>&1 | tee "${WORK_DIR}/logs/production_cache.log"
    else
      echo "[pipeline] [4/4] production cache exists, skipping"
    fi
    PRODUCTION_CACHE_PATH="$PROD_CACHE_RAW"
  fi
fi

# -----------------------------------------------------------------------
# [3c] AURA additivity report (re-vet R2 precondition (ii) / R3; the R2-vs-R19
# disagreement resolved as WIRED).
#
# AURA's one structural assumption is that per-Linear KL contributions ADD.
# This stage turns that assumption into a per-artifact NUMBER —
#   residual = measured_end_KL(assignment) - sum_i predicted_dloss_i
# with an honest stderr (exact per-probe when the cost rows carry
# x2_per_probe, else the independence lower bound) — instead of a two-model
# memory. It is a REPORT: it never fails the run, and it never changes an
# allocation.
#
# WHY IT IS HERE AND NOT AT [2d]. The residual needs a concrete assignment,
# and none exists until the allocator has run (and, under
# validated-surrogate, until the frontier has been selected). This is the
# first point where layer_config.json is final under both selection modes.
#
# AURA_ADDITIVITY_GATE:
#   measure (DEFAULT, ruled 2026-07-30) — report from a measured KL this run
#                    already produced when there is one (validated-surrogate's
#                    frontier JSON, free), and otherwise run ONE bounded KL
#                    measurement of the final assignment against the SAME
#                    format-menu dW cache AURA costed on (AURA_COST_NSAMPLES x
#                    AURA_COST_SEQLEN). Robert's ruling on the R2 residue: the
#                    wiring's weak spot was that under SELECTION_MODE=surrogate
#                    an artifact carried a *prediction* and no residual, so
#                    AURA's one structural assumption stayed a two-model memory
#                    instead of a per-artifact number. Every AURA-default run
#                    now performs the eval and every artifact carries a real
#                    residual. It is still a REPORT — non-blocking, and it never
#                    changes an allocation.
#   auto           — the pre-ruling behaviour: report only from a measurement
#                    the run already made; otherwise record the predicted sum
#                    with measured_kl null and a status saying why. Zero added
#                    GPU.
#   0              — off.
# -----------------------------------------------------------------------
: "${AURA_ADDITIVITY_GATE:=measure}"
if [[ "$COST_MODE" == "aura" && "$AURA_ADDITIVITY_GATE" != "0" \
   && "$AURA_ADDITIVITY_GATE" != "false" && "$AURA_ADDITIVITY_GATE" != "off" ]]; then
  AURA_ADDITIVITY_JSON="${WORK_DIR}/artifacts/aura_additivity.json"
  AURA_MEASURED_KL=""
  AURA_MEASURED_KL_STDERR="0"
  AURA_MEASURED_SOURCE="none"
  AURA_FRONTIER_JSON="${WORK_DIR}/artifacts/validated_frontier_kl.json"
  if [[ -f "$AURA_FRONTIER_JSON" ]]; then
    read -r AURA_MEASURED_KL AURA_MEASURED_KL_STDERR <<<"$(
      python3 - "$AURA_FRONTIER_JSON" "${WORK_DIR}/artifacts/validated_frontier_selection.json" <<'PY'
import json
import sys
from pathlib import Path

results = json.loads(Path(sys.argv[1]).read_text()).get("results", [])
label = None
sel = Path(sys.argv[2])
if sel.is_file():
    payload = json.loads(sel.read_text())
    label = (payload.get("selected") or {}).get("label") or payload.get("label")
row = None
for item in results:
    if label is not None and item.get("label") == label:
        row = item
        break
if row is None and results:
    row = min(results, key=lambda r: float(r.get("kl_mean", r.get("kl", 1e9))))
if row is None:
    print("")
else:
    kl = row.get("kl_mean", row.get("kl"))
    print(f"{float(kl)} {float(row.get('kl_stderr', 0.0) or 0.0)}")
PY
    )"
    [[ -n "$AURA_MEASURED_KL" ]] && AURA_MEASURED_SOURCE="validated_frontier_kl.json"
  fi
  if [[ -z "$AURA_MEASURED_KL" && "$AURA_ADDITIVITY_GATE" == "measure" ]]; then
    AURA_ADDITIVITY_KL_JSON="${WORK_DIR}/artifacts/aura_additivity_kl.json"
    if [[ ! -f "$AURA_ADDITIVITY_KL_JSON" ]]; then
      echo "[pipeline] [3c] measuring end-KL for the additivity report (AURA_ADDITIVITY_GATE=measure) ..."
      python3 -m prismaquant.validate_assignments_kl \
        --model "$MODEL_PATH" \
        --probe "$PROBE_PATH" \
        --costs "$COST_PATH" \
        --base-assignment "${WORK_DIR}/artifacts/layer_config.json" \
        --assignment "selected=${WORK_DIR}/artifacts/layer_config.json" \
        --formats "$FORMATS" \
        --dataset "$DATASET" \
        --n-calib-samples "$AURA_COST_NSAMPLES" \
        --calib-seqlen "$AURA_COST_SEQLEN" \
        --calib-seed "$AURA_COST_CALIB_SEED" \
        --kl-scope full_sequence \
        --assignment-materialization inplace \
        --production-weight-cache "$PRODUCTION_RENDER_COST_CACHE_PATH" \
        --production-cache-dir-override "$PRODUCTION_RENDER_COST_CACHE_DIR" \
        --work-dir "${WORK_DIR}/work/aura_additivity" \
        --dtype bf16 --device "$DEVICE" \
        --output "$AURA_ADDITIVITY_KL_JSON" \
        2>&1 | tee "${WORK_DIR}/logs/aura_additivity_kl.log" || true
    fi
    if [[ -f "$AURA_ADDITIVITY_KL_JSON" ]]; then
      read -r AURA_MEASURED_KL AURA_MEASURED_KL_STDERR <<<"$(
        python3 - "$AURA_ADDITIVITY_KL_JSON" <<'PY'
import json
import sys
from pathlib import Path

results = json.loads(Path(sys.argv[1]).read_text()).get("results", [])
if not results:
    print("")
else:
    row = results[0]
    print(f"{float(row.get('kl_mean', row.get('kl', 0.0)))} "
          f"{float(row.get('kl_stderr', 0.0) or 0.0)}")
PY
      )"
      [[ -n "$AURA_MEASURED_KL" ]] && AURA_MEASURED_SOURCE="aura_additivity_kl.json"
    fi
  fi
  if [[ -n "$AURA_MEASURED_KL" ]]; then
    echo "[pipeline] [3c] AURA additivity report (measured KL from ${AURA_MEASURED_SOURCE}) ..."
    python3 -m prismaquant.aura_additivity_gate \
      --costs "$COST_PATH" \
      --assignment "${WORK_DIR}/artifacts/layer_config.json" \
      --measured-kl "$AURA_MEASURED_KL" \
      --measured-kl-stderr "$AURA_MEASURED_KL_STDERR" \
      --output "$AURA_ADDITIVITY_JSON" \
      2>&1 | tee "${WORK_DIR}/logs/aura_additivity.log" || \
      echo "[pipeline] [3c] WARNING: additivity report failed (non-blocking)"
  else
    echo "[pipeline] [3c] AURA additivity report: no measured end-KL in this run (SELECTION_MODE=${SELECTION_MODE}); recording the predicted sum only. Set AURA_ADDITIVITY_GATE=measure for the residual."
    python3 - "$COST_PATH" "${WORK_DIR}/artifacts/layer_config.json" "$AURA_ADDITIVITY_JSON" <<'PY' || \
      echo "[pipeline] [3c] WARNING: additivity report failed (non-blocking)"
import json
import pickle
import sys

from prismaquant.aura_additivity_gate import additivity_gate
from prismaquant.layer_config import load_assignment

with open(sys.argv[1], "rb") as fh:
    payload = pickle.load(fh)
result = additivity_gate(payload, load_assignment(sys.argv[2]), 0.0)
# measured_kl=0.0 is a placeholder, not a measurement: null it out and say so,
# rather than publish a residual equal to -sum(predicted).
result["measured_kl"] = None
result["residual"] = None
result["residual_over_measured"] = None
result["residual_z"] = None
result["status"] = "no_measured_end_kl_in_this_run"
with open(sys.argv[3], "w") as fh:
    json.dump(result, fh, indent=1)
print(json.dumps(result, indent=1))
PY
  fi
  # Stamp the trust-region number into the cost table's provenance so every
  # artifact derived from it carries it (the R2 provenance contract).
  if [[ -f "$AURA_ADDITIVITY_JSON" ]]; then
    python3 - "$COST_PATH" "$AURA_ADDITIVITY_JSON" "$AURA_MEASURED_SOURCE" <<'PY' || \
      echo "[pipeline] [3c] WARNING: could not stamp additivity provenance (non-blocking)"
import json
import pickle
import sys

cost_path, report_path, source = sys.argv[1:4]
with open(cost_path, "rb") as fh:
    payload = pickle.load(fh)
report = json.loads(open(report_path).read())
prov = dict(payload.get("provenance", {}) or {})
prov["additivity"] = {
    key: report.get(key)
    for key in (
        "measured_kl", "measured_kl_stderr", "predicted_sum",
        "predicted_stderr", "stderr_method", "residual",
        "residual_over_measured", "residual_z", "n_covered", "n_zero_cost",
        "status",
    )
}
prov["additivity"]["measured_kl_source"] = source
payload["provenance"] = prov
with open(cost_path, "wb") as fh:
    pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
add = prov["additivity"]
print(f"[pipeline] [3c] additivity stamped into {cost_path}: "
      f"predicted_sum={add['predicted_sum']:.6g} "
      f"+-{add['predicted_stderr']:.3g} ({add['stderr_method']}) "
      f"residual={add['residual']} z={add['residual_z']}")
PY
  fi
fi

# -----------------------------------------------------------------------
# [3d] Discovery-walker export gate (R5; §8.8 +
# docs/design/model_coverage_ledgers.md).
#
# The pipeline's decision universe used to be defined by the pipeline's own
# enumeration — the exact hole `attn.wo_a` shipped through (17.9% of DSv4
# decode read traffic, fed by a grouped einsum on a module class the probe
# skips, never an allocator decision, every coverage stat still green). This
# stage re-derives the universe from the object itself: module tree plus one
# FakeTensorMode forward over a META load (no GPU, no weight I/O), every node
# claimed decide/pin/exclude by the profile's walk_claim_rules().
#
# FAIL-CLOSED: exit 2 refuses the run before ANY export lane starts. A refusal
# is decided from STRUCTURED fields only (failure kinds; per-node op/equation/
# module lists in artifacts/model_walk.json) — never prose. An explicit
# PRISMAQUANT_WALK_GATE_OVERRIDE=<reason> excuses TRACE INCOMPLETENESS ONLY
# (e.g. DSv4's data-dependent scalar aborting the fake trace); it is stamped
# into the report and NEVER excuses an unclaimed node — claims are pinned with
# reasons in ModelProfile.walk_claim_rules(), not waived at export time.
# -----------------------------------------------------------------------
: "${WALK_GATE_SEQLEN:=8}"
# auto->fake: the fake trace is the intended cheap path; an abort becomes a
# structured incomplete-trace refusal (override env or --execution real via a
# manual gate run), never a silent pass.
: "${WALK_GATE_EXECUTION:=fake}"
: "${PRISMAQUANT_WALK_GATE_OVERRIDE:=}"
MODEL_WALK_REPORT="${WORK_DIR}/artifacts/model_walk.json"
echo "[pipeline] [3d] discovery-walker export gate ..."
WALK_GATE_ARGS=(
  python3 -m prismaquant.model_walk
  --model "$MODEL_PATH"
  --execution "$WALK_GATE_EXECUTION"
  --seq-len "$WALK_GATE_SEQLEN"
  --output "$MODEL_WALK_REPORT"
)
[[ -n "$PRISMAQUANT_WALK_GATE_OVERRIDE" ]] && WALK_GATE_ARGS+=(
  --override-reason "$PRISMAQUANT_WALK_GATE_OVERRIDE"
)
"${WALK_GATE_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/model_walk.log"

if [[ "$EXPORT_CONTAINER" == "tessera" ]]; then
  # Tessera lane: one blob per vLLM module on the Tessera wire, served by
  # Tessera's OWN vLLM plugin (package `tessera.serving`, entry point under
  # vllm.general_plugins, quant_method "tessera"). There is no enable flag --
  # the checkpoint selects the plugin. Scoped targets additionally bind the
  # exact image and execution mode rather than inheriting wrapper defaults.
  #
  # TWO CALLS OUT, ZERO CODECS IN. The layer_config -> plan translation and
  # the encode both live in the Tessera repository and are NAMED here, never
  # copied: `plan_from_layer_config.py` is the only place the
  # `TESSERA_<BASE>_K<arity>_R<rung>` spelling is turned into the exporter's
  # (grid, q256), and `export_tessera_serving.py` is the only place the wire
  # is written. A second copy of either in this repository would be a second
  # place a wire recipe can drift, which is exactly what the producer/consumer
  # boundary exists to prevent. The lane preflight above has already refused
  # if TESSERA_REPO does not hold both.
  # Re-read the allocation's scope and actual source header dimensions even
  # when an old plan exists: a cached plan is not an admission receipt.
  TESSERA_BUILD_JSON="${WORK_DIR}/artifacts/tessera_build.json"
  # The priced inputs, threaded to the preflight (which refuses when the
  # allocation declares a requirement they do not satisfy) and to the exporter
  # (which consumes them). Built inside the arm so the block is self-contained.
  TESSERA_PRICED_INPUT_ARGS=()
  if [[ -n "${TESSERA_HESSIAN:-}" ]]; then
    TESSERA_PRICED_INPUT_ARGS+=(--hessian "$TESSERA_HESSIAN")
  fi
  if [[ -n "${TESSERA_INPUT_SCALES:-}" ]]; then
    TESSERA_PRICED_INPUT_ARGS+=(--input-scales "$TESSERA_INPUT_SCALES")
  fi
  if ! python3 -m prismaquant.tessera_export_lane --model "$MODEL_PATH" \
      --assignment "${WORK_DIR}/artifacts/layer_config.json" \
      --write-build-json "$TESSERA_BUILD_JSON" \
      --target-profile "$TARGET_PROFILE_RESOLVED" "${TESSERA_SCOPE_ARGS[@]}" \
      "${TESSERA_PRICED_INPUT_ARGS[@]}"; then
    exit 2
  fi
  TESSERA_PLAN="${WORK_DIR}/artifacts/tessera_plan.json"
  # The allocator always rewrites its recipe. A path or a newly generated
  # build anchor cannot establish which allocation an existing plan translated.
  TESSERA_ASSIGNMENT_DIGEST=$(sha256sum "${WORK_DIR}/artifacts/layer_config.json")
  TESSERA_ASSIGNMENT_DIGEST=${TESSERA_ASSIGNMENT_DIGEST%% *}
  require_stage_settings "$TESSERA_PLAN" tessera-plan \
    "ASSIGNMENT_DIGEST=$TESSERA_ASSIGNMENT_DIGEST"
  if [[ ! -f "$TESSERA_PLAN" ]]; then
    echo "[pipeline] [4/4] translating layer_config.json -> Tessera plan (cover=${TESSERA_PLAN_COVER}) ..."
    # Write-then-rename: a crashed translation must not leave a partial plan
    # that the skip-gate above then trusts.
    python3 "${TESSERA_REPO%/}/experiments/plan_from_layer_config.py" \
      "${WORK_DIR}/artifacts/layer_config.json" \
      "$MODEL_PATH" \
      "${TESSERA_PLAN}.tmp" \
      --cover "$TESSERA_PLAN_COVER" \
      --prismaquant "$PIPELINE_SCRIPT_DIR/.." \
      2>&1 | tee "${WORK_DIR}/logs/tessera_plan.log"
    mv "${TESSERA_PLAN}.tmp" "$TESSERA_PLAN"
    # The translator writes `<out>.provenance.json` beside the plan: the
    # source path, the allocation's own __prismaquant__ block, the coverage
    # decision, and the per-unit shape/rung/wire-bytes table an export is
    # checked against. It moves with the plan, not after it. An explicit `if`
    # rather than `[[ ... ]] &&` because a false test would be this block's
    # exit status under `set -e`.
    if [[ -f "${TESSERA_PLAN}.tmp.provenance.json" ]]; then
      mv "${TESSERA_PLAN}.tmp.provenance.json" "${TESSERA_PLAN}.provenance.json"
    fi
  else
    echo "[pipeline] [4/4] Tessera plan exists, skipping"
  fi

  echo "[pipeline] [4/4] exporting to the Tessera wire ..."
  # The same priced inputs the preflight just validated, handed to the encode:
  # the Hessian that shaped the priced bytes and the static activation scales
  # the W4A4 costs were scored under. Omitting them here while the preflight
  # accepted an H-free/scale-free allocation is the weights-only lane and
  # correct; omitting them on an H-aware allocation is unreachable -- the
  # preflight exits 2 above before this line runs.
  python3 "${TESSERA_REPO%/}/experiments/export_tessera_serving.py" \
    "$MODEL_PATH" "${WORK_DIR}/exported" \
    --plan-json "$TESSERA_PLAN" \
    --device "$EXPORT_DEVICE" \
    "${TESSERA_PRICED_INPUT_ARGS[@]}" \
    2>&1 | tee "${WORK_DIR}/logs/export.log"

  echo
  echo "[pipeline] done."
  echo "  Artifact: ${WORK_DIR}/exported"
  # The build lane OPENS the ship record; the serve lane CLOSES it (R13).
  #
  # It did not, on this lane, until 2026-09-03: Tessera's exporter has no
  # concept of a PrismaQuant shipcard and this arm exits ~130 lines above the
  # driver's shipcard block, so every gate lane_specs/tessera.json declares was
  # enforced by nothing (RobTand/prismaquant#119). `lane_shipcard open` opens a
  # record whose slots ARE this lane's declared gates -- including
  # `route.census`, principle 12's second leg, which carried no slot at all --
  # so an un-run gate is now an unfilled slot on a real card that
  # tools/publish_artifact.py refuses on, instead of a sentence in a JSON file.
  #
  # The gates are still NAMED, not RUN, here: each needs a fresh vLLM container
  # with the pinned plugin editable-installed, and both residencies, because
  # the two modes decode the same bytes by different paths. Building that
  # runner is R16's open half and stays with #119.
  #
  # Skip-if-exists like every other stage in this driver, and for the same
  # reason plus one: re-opening a card DISCARDS every slot the serve lane has
  # already filled, so `lane_shipcard open` refuses without --overwrite. A
  # re-run over a completed build would otherwise die here on a refusal whose
  # message ("unpublishable until one exists") is false -- one exists.
  if [[ -f "${WORK_DIR}/exported/shipcard.json" ]]; then
    echo "  Ship record: ${WORK_DIR}/exported/shipcard.json exists, kept (re-open would discard filled slots)"
  elif ! python3 -m prismaquant.lane_shipcard open \
         --lane tessera --artifact "${WORK_DIR}/exported" \
         --build-json "$TESSERA_BUILD_JSON"; then
    echo "[pipeline] ERROR: EXPORT_CONTAINER=tessera: could not open the Tessera ship record. The artifact is unpublishable without one, and writing a base card by hand would omit this lane's own gates, so the export is not done until this succeeds." >&2
    exit 2
  fi
  # Print a shell-safe recipe for the same target admission just checked.
  # Legacy context-free callers retain the wrapper's image/mode defaults.
  TESSERA_PRINT_SERVE=("TESSERA_SERVE_MODE=${TESSERA_SERVE_MODE}" "TS=${TESSERA_REPO%/}")
  if [[ -n "${TESSERA_RUNTIME_IMAGE:-}" ]]; then
    TESSERA_PRINT_SERVE+=("IMAGE=$TESSERA_RUNTIME_IMAGE")
    case "$TESSERA_EXECUTION_MODE" in
      eager) TESSERA_PRINT_SERVE+=(TESSERA_LANE_EAGER=1) ;;
      compiled) TESSERA_PRINT_SERVE+=(TESSERA_LANE_EAGER=0) ;;
      *) echo "[pipeline] ERROR: invalid scoped Tessera execution mode" >&2; exit 2 ;;
    esac
  fi
  TESSERA_PRINT_SERVE+=(bash "${TESSERA_REPO%/}/experiments/tessera_plugin_served.sh"
    "${WORK_DIR}/exported" '<arm>' "$TESSERA_SERVE_MODE")
  printf '  Serve:      '
  printf ' %q' "${TESSERA_PRINT_SERVE[@]}"
  printf '\n'
  echo "  Run the route census inside the same verified runtime image; keep its complete raw JSON."
  TESSERA_PRINT_CENSUS=("TESSERA_SERVE_MODE=${TESSERA_SERVE_MODE}" python3
    "${TESSERA_REPO%/}/tools/tessera_route_census.py" "${WORK_DIR}/exported"
    '<raw-census.json>' --runtime-image "${TESSERA_RUNTIME_IMAGE:-<verified-image@sha256:digest>}")
  if [[ "${TESSERA_EXECUTION_MODE:-}" == "compiled" ]]; then
    TESSERA_PRINT_CENSUS+=(--compiled)
    echo "  Note: the producer's combined dense compiled trace cannot prove per-regime route agreement."
  fi
  printf '  Route census:'
  printf ' %q' "${TESSERA_PRINT_CENSUS[@]}"
  printf '\n'
  printf '  Close census:'
  printf ' %q' python3 -m prismaquant.shipcard_cli fill-route-census \
    "${WORK_DIR}/exported/shipcard.json" --census '<raw-census.json>' \
    --layer-config "${WORK_DIR}/artifacts/layer_config.json" --model-dir "${WORK_DIR}/exported"
  printf '\n'
  echo "  Verify:      python -m prismaquant.shipcard_cli verify ${WORK_DIR}/exported/shipcard.json"
  exit 0
fi

if [[ "$EXPORT_CONTAINER" == "gguf" ]]; then
  # GGUF lane: one artifact serves llama.cpp natively and vLLM via the
  # GGUF path. Requires TARGET_PROFILE=gguf at allocation time (the
  # exporter hard-fails on non-GGUF formats). The skeleton (metadata +
  # tokenizer) comes from llama.cpp's own converter; we requantize bytes.
  : "${LLAMA_CPP_DIR:=/home/rob/dq-runs/llama.cpp}"
  : "${GGUF_SKELETON:=${WORK_DIR}/artifacts/skeleton.gguf}"
  : "${GGUF_TOKEN_EMBEDDING_FORMAT:=}"
  : "${GGUF_OUTPUT_FORMAT:=}"
  require_stage_settings "$GGUF_SKELETON" gguf-skeleton
  if [[ ! -f "$GGUF_SKELETON" ]]; then
    echo "[pipeline] [4/4] building GGUF skeleton (convert_hf_to_gguf, bf16) ..."
    # Write-then-rename: convert_hf_to_gguf writes in place, so a crashed
    # conversion must not leave a truncated file the skip-gate trusts.
    python3 "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$MODEL_PATH" \
      --outtype bf16 --outfile "${GGUF_SKELETON}.tmp" \
      2>&1 | tee "${WORK_DIR}/logs/gguf_skeleton.log"
    mv "${GGUF_SKELETON}.tmp" "$GGUF_SKELETON"
  else
    echo "[pipeline] [4/4] GGUF skeleton exists, skipping"
  fi
  echo "[pipeline] [4/4] exporting to GGUF ..."
  GGUF_EXPORT_ARGS=(
    python3 -m prismaquant.export_gguf
    --skeleton "$GGUF_SKELETON"
    --layer-config "${WORK_DIR}/artifacts/layer_config.json"
    --out "${WORK_DIR}/exported.gguf"
    --device "$EXPORT_DEVICE"
  )
  # Keep export-side imatrix in lockstep with the cost measurement
  # (PRISMAQUANT_GGUF_IMATRIX, default on): measured cost and shipped
  # bytes must be the same rendering. The truthiness parse MUST match
  # measure_quant_cost._gguf_imatrix_enabled (set-but-empty = default on;
  # 0/false/no/off in any case = off), or the one flag whose job is
  # lockstep silently splits the two sides.
  _gguf_imatrix="${PRISMAQUANT_GGUF_IMATRIX:-1}"
  _gguf_imatrix="$(echo "$_gguf_imatrix" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
  [[ -z "$_gguf_imatrix" ]] && _gguf_imatrix=1
  case "$_gguf_imatrix" in
    0|false|no|off) ;;
    *) GGUF_EXPORT_ARGS+=(--imatrix-from-act-cache "${WORK_DIR}/act") ;;
  esac
  [[ -n "$GGUF_TOKEN_EMBEDDING_FORMAT" ]] && \
    GGUF_EXPORT_ARGS+=(--token-embedding-format "$GGUF_TOKEN_EMBEDDING_FORMAT")
  [[ -n "$GGUF_OUTPUT_FORMAT" ]] && \
    GGUF_EXPORT_ARGS+=(--output-format "$GGUF_OUTPUT_FORMAT")
  "${GGUF_EXPORT_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/export.log"

  # Load + greedy-generate smoke on the actual serving runtime (the
  # validate_native_export analog for this lane). A PPL/p99-NLL ship gate
  # (validate_quantized_model analog) is still open work — see
  # docs/lanes/gguf.md; this gate only proves the artifact loads and
  # produces tokens.
  LLAMA_COMPLETION="$LLAMA_CPP_DIR/build/bin/llama-completion"
  if [[ -x "$LLAMA_COMPLETION" ]]; then
    echo "[pipeline] [4/4] llama.cpp load+generate smoke ..."
    SMOKE_OUT=$("$LLAMA_COMPLETION" -m "${WORK_DIR}/exported.gguf" \
      -p "The quick brown fox" -n 16 --temp 0 -ngl 99 --no-display-prompt \
      < /dev/null 2>"${WORK_DIR}/logs/gguf_smoke.err") || {
      echo "[pipeline] ERROR: llama.cpp failed to load/generate from the exported artifact; see ${WORK_DIR}/logs/gguf_smoke.err" >&2
      exit 1
    }
    if [[ -z "${SMOKE_OUT//[[:space:]]/}" ]]; then
      echo "[pipeline] ERROR: llama.cpp generated no tokens from the exported artifact" >&2
      exit 1
    fi
    echo "[pipeline] [4/4] smoke output: ${SMOKE_OUT:0:120}"
  else
    echo "[pipeline] WARNING: $LLAMA_COMPLETION not built — skipping load+generate smoke (build with: cmake --build $LLAMA_CPP_DIR/build --target llama-completion)"
  fi

  echo
  echo "[pipeline] done."
  echo "  Artifact: ${WORK_DIR}/exported.gguf"
  echo "  Serve (llama.cpp): $LLAMA_CPP_DIR/build/bin/llama-server -m ${WORK_DIR}/exported.gguf -ngl 99"
  echo "  KL harness:        llama-perplexity --kl-divergence-base <base_logits> --kl-divergence"
  exit 0
fi


echo "[pipeline] [4/4] exporting to compressed-tensors ..."
EXPORT_ARGS=(
  python3 -m prismaquant.export_native_compressed
  --model "$MODEL_PATH"
  --layer-config "${WORK_DIR}/artifacts/layer_config.json"
  --output "${WORK_DIR}/exported"
  --device "$EXPORT_DEVICE"
  --shard-bytes "$EXPORT_SHARD_BYTES"
  --activation-cache-dir "${WORK_DIR}/act"
  "${EXPORT_PIN_ARGS[@]}"
)
case "$EXPORT_GPTQ" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-gptq) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--gptq) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_GPTQ must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
case "$EXPORT_SCALE_SWEEP" in
  0|false|False|FALSE|no|No|NO) EXPORT_ARGS+=(--no-scale-sweep) ;;
  1|true|True|TRUE|yes|Yes|YES) EXPORT_ARGS+=(--scale-sweep) ;;
  auto|"") ;;
  *)
    echo "[pipeline] ERROR: EXPORT_SCALE_SWEEP must be auto, 0, or 1" >&2
    exit 2
    ;;
esac
if [[ -n "$PRODUCTION_CACHE_PATH" ]]; then
  # --production-cache-prefetch=require (re-vet R24/D8): a total prefetch miss
  # used to be invisible — the helper returned 0 on every failure path and the
  # caller only logged when it prefetched something — so the export silently
  # went NVMe-bound tensor by tensor. `require` on the native lane mirrors
  # VALIDATED_SOURCE_PREFETCH=require and production_weight_cache's own
  # require mode; the CB/GGUF lanes never read a production cache at all.
  EXPORT_ARGS+=(
    --production-weight-cache "$PRODUCTION_CACHE_PATH"
    --production-cache-dir-override "$PROD_CACHE_DIR"
    --production-cache-lru-gb "$PRODUCTION_CACHE_LRU_GB"
    --production-cache-prefetch "$EXPORT_PRODUCTION_CACHE_PREFETCH"
    --production-cache-prefetch-workers "$PRODUCTION_CACHE_PREFETCH_WORKERS"
  )
fi
"${EXPORT_ARGS[@]}" 2>&1 | tee "${WORK_DIR}/logs/export.log"

echo
echo "[pipeline] done."
echo "  Artifact: ${WORK_DIR}/exported"
# The build lane OPENS the ship record; the serve lane must CLOSE it (R13).
# Printing the open slots here is the whole point of a refusal contract: the
# run ends by naming what has NOT been measured yet, instead of echoing a
# command and implying the artifact is done.
SHIPCARD_JSON="${WORK_DIR}/exported/shipcard.json"
if [[ -f "$SHIPCARD_JSON" ]]; then
  python3 -m prismaquant.shipcard_cli show "$SHIPCARD_JSON" || true
else
  echo "  WARNING: no shipcard at ${SHIPCARD_JSON} — the export did not open a ship record."
fi
