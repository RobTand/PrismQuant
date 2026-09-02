#!/usr/bin/env bash
# ============================================================================
# encode_hy3_mtp_cb.sh — CB-quantise the Hy3 MTP draft (model.layers.80.*)
# ============================================================================
# Robert's rule: "always include MTP when it's available". The Hy3 body ships
# CB (prod-hy3-nvfp4cb-2p9) but its bf16 MTP draft (7.5 GB) OOMs next to
# ~102 GiB of weights on the 128 GB Spark. A draft's weights CANNOT change
# outputs (spec decode is exact via rejection sampling) — only the acceptance
# rate — so a hand-chosen modal-rung policy is fine and documented as such:
#   * routed experts  -> the body's modal fp4 expert rung (NVFP4_CB_K18)
#   * shared + attn    -> FP8_CB_K32
#   * norms/eh_proj/router/expert_bias  -> bf16/f32 (carried through)
#
# Two steps: (1) build the MTP layer_config + UNIFORM col-weights (no imatrix);
# (2) stream-encode ONLY model.layers.80.* via export_nvfp4_cb_streaming's
# --subset-prefix (so the ~550 GB body is NOT re-copied as bf16 passthrough).
# lattice codebooks (default) are deterministic per rung -> the MTP's K18/K32
# tables are BIT-IDENTICAL to the body's, which the merge step relies on.
#
# GPU step. Run when the box is free; the exporter self-caps at 0.75 of the
# unified pool. Output goes to $WORK/mtp_cb; merge with the body afterwards via
# scripts/merge_hy3_mtp_reshard.py.
# ============================================================================
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SOURCE="${SOURCE:-/home/rob/dq-runs/hy3-prod/source}"
WORK="${WORK:-/home/rob/dq-runs/prod-hy3-nvfp4cb-2p9}"
BODY_LC="${BODY_LC:-${WORK}/artifacts/layer_config.json}"
OUT="${OUT:-${WORK}/mtp_cb}"
INPUTS="${INPUTS:-${WORK}/mtp_inputs}"
MTP_LAYER="${MTP_LAYER:-80}"
PY="${PY:-/home/rob/dq-runs/venvs/prismaquant-cu130/bin/python}"

for p in "$SOURCE/config.json" "$BODY_LC"; do
  [[ -e "$p" ]] || { echo "FATAL: missing $p" >&2; exit 2; }
done
mkdir -p "$INPUTS" "$OUT"

echo "============================================================================"
echo "Hy3 MTP CB encode — layer ${MTP_LAYER}  (draft; modal-rung policy)"
echo "  SOURCE=$SOURCE"
echo "  BODY_LC=$BODY_LC"
echo "  OUT=$OUT"
echo "  start: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================================"

echo "## step 1/2: build MTP layer_config + uniform col-weights"
PYTHONPATH="$REPO" "$PY" "$REPO/scripts/build_hy3_mtp_cb_inputs.py" \
  --source-dir "$SOURCE" \
  --body-layer-config "$BODY_LC" \
  --out-dir "$INPUTS" \
  --mtp-layer "$MTP_LAYER"

LC="$INPUTS/mtp_layer_config.json"
CW="$INPUTS/mtp_col_weights.pkl"
[[ -s "$LC" && -s "$CW" ]] || { echo "FATAL: inputs not written" >&2; exit 2; }

echo
echo "## step 2/2: stream-encode model.layers.${MTP_LAYER}.* (CB) -> $OUT"
PYTHONPATH="$REPO" "$PY" -m prismaquant.export_nvfp4_cb_streaming \
  --model-dir "$SOURCE" \
  --layer-config "$LC" \
  --out "$OUT" \
  --col-weights "$CW" \
  --codebook-source lattice \
  --scale-coding two_tier \
  --subset-prefix "model.layers.${MTP_LAYER}." \
  --device cuda

echo
echo "##  MTP CB ENCODE DONE -> $OUT"
echo "##  tensors:";  ls -la "$OUT" 2>/dev/null || true
echo "##  Next: scripts/merge_hy3_mtp_reshard.py \\"
echo "##          --main-dir ${WORK}/exported_nvfp4_cb --mtp-dir $OUT \\"
echo "##          --out ${WORK}/ship_mtp"
echo "  end: $(date '+%Y-%m-%d %H:%M:%S')"
