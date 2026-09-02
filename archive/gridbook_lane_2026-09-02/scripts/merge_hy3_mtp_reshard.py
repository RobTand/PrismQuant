#!/usr/bin/env python3
"""Merge the CB MTP sidecar into the Hy3 body artifact -> a sharded HF dir.

Assembles the shippable artifact from two inputs:

  * ``--main-dir``: the body CB artifact (single ``model.safetensors`` + its
    ``quant_config.json`` / ``cb_codebooks.pqcb`` / config / tokenizer). Its
    ``model.layers.80.*`` are the OLD bf16 MTP passthrough (7.5 GB).
  * ``--mtp-dir``: the CB MTP export from ``encode_hy3_mtp_cb.sh`` (only
    ``model.layers.80.*``: the CB tensors + bf16 glue + its own quant_config /
    codebooks).

The output ``--out`` dir:
  1. drops the OLD bf16 ``model.layers.80.*`` EXCEPT the bf16 glue
     (enorm/hnorm/eh_proj/final_layernorm, the block's norms, q_norm/k_norm,
     router gate, expert_bias) — kept from MAIN; ``embed_tokens``/``shared_head``
     under layer 80 are dropped entirely (vLLM reuses the main embed/lm_head);
  2. adds the CB MTP tensors (``.cb_qweight`` / ``.weight_scale``) from MTP;
  3. merges ``quant_config.json`` (config_groups by scheme signature + ignore),
     asserting no target collides;
  4. asserts the two ``cb_codebooks.pqcb`` sidecars AGREE bit-for-bit on shared
     codebook names (both lattice -> identical per rung), then unions them;
  5. copies config/tokenizer/chat-template from MAIN (asserting the config still
     declares the MTP: ``num_nextn_predict_layers > 0``);
  6. writes shards <= ``--shard-gb`` + ``model.safetensors.index.json``.

Streaming byte-copy per tensor (torch-free, RAM ~ the copy buffer): tensor bytes
are copied verbatim from whichever source file owns them. Prints a byte
accounting (dropped / added / kept / total). CPU-only; no GPU, no torch.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

BUF = 64 * 1024 * 1024

# A layer-80 tensor is DROPPED from MAIN iff it is a CB-replaced role or the
# MTP's own (unused) embed/head; everything else under layer 80 is bf16 glue and
# is KEPT. Negative definition so a future norm/bias is carried through by
# default rather than silently dropped.
_DROP_ROLE_MARKERS = (
    ".self_attn.q_proj.", ".self_attn.k_proj.", ".self_attn.v_proj.",
    ".self_attn.o_proj.", ".mlp.shared_mlp.", ".mlp.experts.",
    ".embed_tokens", ".shared_head",
)


def _read_header(path: Path) -> tuple[dict, int]:
    """Return (header dict without __metadata__, data_base offset)."""
    with open(path, "rb") as f:
        (hlen,) = struct.unpack("<Q", f.read(8))
        hdr = json.loads(f.read(hlen))
    hdr.pop("__metadata__", None)
    return hdr, 8 + hlen


def _tensor_bytes(path: Path, data_base: int, entry: dict) -> bytes:
    o0, o1 = entry["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_base + o0)
        return f.read(o1 - o0)


def _copy_tensor(src: Path, data_base: int, entry: dict, out) -> int:
    o0, o1 = entry["data_offsets"]
    left = o1 - o0
    with open(src, "rb") as f:
        f.seek(data_base + o0)
        while left:
            chunk = f.read(min(BUF, left))
            if not chunk:
                raise IOError(f"short read {left} bytes remaining")
            out.write(chunk)
            left -= len(chunk)
    return o1 - o0


def _entry_size(entry: dict) -> int:
    return entry["data_offsets"][1] - entry["data_offsets"][0]


def _merge_codebooks(main_cb: Path, mtp_cb: Path, out_cb: Path) -> None:
    """Union the two lattice codebook sidecars, asserting shared names agree
    bit-for-bit. Torch-free: compare/copy raw tensor byte ranges."""
    mh, mb = _read_header(main_cb)
    th, tb = _read_header(mtp_cb)
    shared = sorted(set(mh) & set(th))
    for name in shared:
        if _tensor_bytes(main_cb, mb, mh[name]) != _tensor_bytes(mtp_cb, tb, th[name]):
            raise SystemExit(
                f"FATAL: codebook '{name}' differs between main and MTP sidecars "
                "— lattice rungs must be bit-identical; refusing to merge")
    # Union: main's tensors, then any MTP-only tensors (re-laid-out contiguously).
    names = list(mh) + [n for n in th if n not in mh]
    src_of = {n: (main_cb, mb, mh[n]) for n in mh}
    for n in th:
        src_of.setdefault(n, (mtp_cb, tb, th[n]))
    header: dict = {}
    off = 0
    for n in names:
        e = src_of[n][2]
        sz = _entry_size(e)
        header[n] = {"dtype": e["dtype"], "shape": e["shape"],
                     "data_offsets": [off, off + sz]}
        off += sz
    header["__metadata__"] = {"format": "pt"}
    hb = json.dumps(header, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    with open(out_cb, "wb") as out:
        out.write(struct.pack("<Q", len(hb)))
        out.write(hb)
        for n in names:
            src, base, e = src_of[n]
            _copy_tensor(src, base, e, out)
    print(f"[merge] codebooks: {len(mh)} main + {len(th) - len(shared)} MTP-only "
          f"= {len(names)} ({len(shared)} shared, all bit-identical)")


def _merge_quant_config(main_qc: dict, mtp_qc: dict, mtp_layer: int,
                        merge_note: dict) -> dict:
    for key in ("quant_method", "format"):
        if main_qc.get(key) != mtp_qc.get(key):
            raise SystemExit(
                f"FATAL: quant_config '{key}' differs "
                f"({main_qc.get(key)!r} vs {mtp_qc.get(key)!r})")
    if main_qc.get("codebook_file") != mtp_qc.get("codebook_file"):
        raise SystemExit("FATAL: quant_config codebook_file differs")

    # Merge config_groups by their FULL non-target signature; union targets.
    # The signature (and the carried dict) is the whole group minus "targets":
    # CB groups keep format+scheme, stock compressed-tensors groups keep their
    # CT vocabulary (weights/input_activations/format) — an earlier version
    # rebuilt groups as {targets, format, scheme} and silently emitted stock
    # groups with "scheme": null, which the plugin then treated as a broken CB
    # group and served UNQUANTIZED (first joint-menu merge, 2026-07-20).
    def sig(g):
        return json.dumps({k: v for k, v in g.items() if k != "targets"},
                          sort_keys=True, separators=(",", ":"))
    by_sig: dict[str, dict] = {}
    target_owner: dict[str, str] = {}
    for src, qc in (("main", main_qc), ("mtp", mtp_qc)):
        for g in qc.get("config_groups", {}).values():
            s = sig(g)
            grp = by_sig.setdefault(
                s, {"body": {k: v for k, v in g.items() if k != "targets"},
                    "targets": set()})
            for t in g.get("targets", []):
                if t in target_owner and target_owner[t] != s:
                    raise SystemExit(
                        f"FATAL: target '{t}' assigned to two different "
                        "schemes across main/MTP configs")
                target_owner[t] = s
                grp["targets"].add(t)
    config_groups = {}
    for i, s in enumerate(sorted(by_sig)):
        grp = by_sig[s]
        config_groups[f"group_{i}"] = {
            "targets": sorted(grp["targets"]), **grp["body"]}

    # ignore: main's minus any layer-80 entries (those roles are CB now), plus
    # the MTP's ignore (its bf16-glue Linears: eh_proj, router gate).
    pref = f"model.layers.{mtp_layer}."
    ignore = {ig for ig in main_qc.get("ignore", []) if not ig.startswith(pref)}
    ignore |= set(mtp_qc.get("ignore", []))

    out = dict(main_qc)
    out["config_groups"] = config_groups
    out["ignore"] = sorted(ignore)
    prov = dict(main_qc.get("provenance", {}))
    prov["mtp_merge"] = merge_note
    out["provenance"] = prov
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main-dir", required=True)
    ap.add_argument("--mtp-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-gb", type=float, default=25.0)
    ap.add_argument("--mtp-layer", type=int, default=80)
    args = ap.parse_args()

    main_dir, mtp_dir, out_dir = Path(args.main_dir), Path(args.mtp_dir), Path(args.out)
    L = args.mtp_layer
    pref = f"model.layers.{L}."
    cap = int(args.shard_gb * 1e9)

    main_st = main_dir / "model.safetensors"
    mtp_st = mtp_dir / "model.safetensors"
    for p in (main_st, mtp_st, main_dir / "quant_config.json",
              mtp_dir / "quant_config.json", main_dir / "cb_codebooks.pqcb",
              mtp_dir / "cb_codebooks.pqcb", main_dir / "config.json"):
        if not p.exists():
            raise SystemExit(f"FATAL: missing {p}")

    main_hdr, main_base = _read_header(main_st)
    mtp_hdr, mtp_base = _read_header(mtp_st)

    # --- classify: which tensors go to the output, and from which file. ---
    # ordered list of (name, entry, src_path, data_base)
    out_tensors: list[tuple[str, dict, Path, int]] = []
    dropped_bytes = kept_body_bytes = kept_glue_bytes = 0

    for name, e in main_hdr.items():
        if not name.startswith(pref):
            out_tensors.append((name, e, main_st, main_base))
            kept_body_bytes += _entry_size(e)
        elif any(m in name for m in _DROP_ROLE_MARKERS):
            dropped_bytes += _entry_size(e)            # old bf16 role / embed/head
        else:
            out_tensors.append((name, e, main_st, main_base))   # bf16 glue
            kept_glue_bytes += _entry_size(e)

    added_bytes = skipped_mtp_bytes = 0
    n_cb = 0
    for name, e in mtp_hdr.items():
        if not name.startswith(pref):
            raise SystemExit(
                f"FATAL: MTP export tensor '{name}' is not under {pref} — the "
                "subset export leaked non-MTP tensors")
        if ".embed_tokens" in name or ".shared_head" in name:
            skipped_mtp_bytes += _entry_size(e)        # dead bytes
        elif name.endswith(".cb_qweight") or name.endswith(".weight_scale"):
            out_tensors.append((name, e, mtp_st, mtp_base))
            added_bytes += _entry_size(e)
            n_cb += 1
        else:
            skipped_mtp_bytes += _entry_size(e)        # MTP's bf16 glue (from main)

    if n_cb == 0:
        raise SystemExit("FATAL: MTP export has no .cb_qweight/.weight_scale "
                         "tensors — nothing to merge")
    names_seen: dict[str, int] = {}
    for i, (name, *_rest) in enumerate(out_tensors):
        if name in names_seen:
            raise SystemExit(f"FATAL: output tensor collision on '{name}'")
        names_seen[name] = i

    # --- merge quant_config + codebooks + config/tokenizer. ---
    out_dir.mkdir(parents=True, exist_ok=True)
    main_qc = json.loads((main_dir / "quant_config.json").read_text())
    mtp_qc = json.loads((mtp_dir / "quant_config.json").read_text())
    merge_note = {
        "mtp_layer": L, "cb_tensors_added": n_cb,
        "mtp_assignment_sha256":
            mtp_qc.get("provenance", {}).get("assignment_sha256"),
        "mtp_git_commit": mtp_qc.get("provenance", {}).get("git_commit"),
    }
    merged_qc = _merge_quant_config(main_qc, mtp_qc, L, merge_note)
    (out_dir / "quant_config.json").write_text(
        json.dumps(merged_qc, indent=2, sort_keys=True))
    _merge_codebooks(main_dir / "cb_codebooks.pqcb", mtp_dir / "cb_codebooks.pqcb",
                     out_dir / "cb_codebooks.pqcb")

    config = json.loads((main_dir / "config.json").read_text())
    if int(config.get("num_nextn_predict_layers", 0) or 0) <= 0:
        raise SystemExit(
            "FATAL: main config.json has num_nextn_predict_layers <= 0 — vLLM "
            "will not build the MTP; a CB MTP would be dead weight")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    for aux in ("tokenizer.json", "tokenizer_config.json", "tokenizer.model",
                "special_tokens_map.json", "generation_config.json",
                "vocab.json", "merges.txt", "chat_template.jinja",
                "chat_template.json"):
        p = main_dir / aux
        if p.exists():
            (out_dir / aux).write_bytes(p.read_bytes())

    # --- greedy source-order packing into shards + streamed byte-copy. ---
    shards: list[list[int]] = [[]]
    used = 0
    for idx, (_n, e, _src, _base) in enumerate(out_tensors):
        sz = _entry_size(e)
        if used and used + sz > cap:
            shards.append([])
            used = 0
        shards[-1].append(idx)
        used += sz
    n = len(shards)
    weight_map: dict[str, str] = {}
    total = 0
    for si, idxs in enumerate(shards, 1):
        fname = f"model-{si:05d}-of-{n:05d}.safetensors"
        sh_hdr: dict = {}
        off = 0
        for idx in idxs:
            name, e, _src, _base = out_tensors[idx]
            sz = _entry_size(e)
            sh_hdr[name] = {"dtype": e["dtype"], "shape": e["shape"],
                            "data_offsets": [off, off + sz]}
            off += sz
            weight_map[name] = fname
        sh_hdr["__metadata__"] = {"format": "pt", "quant_method": "gridbook"}
        hb = json.dumps(sh_hdr, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        with open(out_dir / fname, "wb") as out:
            out.write(struct.pack("<Q", len(hb)))
            out.write(hb)
            for idx in idxs:
                name, e, src, base = out_tensors[idx]
                _copy_tensor(src, base, e, out)
        total += off
        print(f"[merge] {fname}: {len(idxs)} tensors, {off / 1e9:.2f} GB",
              flush=True)

    index = {"metadata": {"total_size": total}, "weight_map": weight_map}
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True))

    print("[merge] ============ byte accounting ============")
    print(f"[merge]   kept   body (non-L{L})     : {kept_body_bytes / 1e9:8.3f} GB")
    print(f"[merge]   kept   L{L} bf16 glue       : {kept_glue_bytes / 1e9:8.3f} GB")
    print(f"[merge]   dropped old bf16 L{L} roles : {dropped_bytes / 1e9:8.3f} GB")
    print(f"[merge]   added  CB MTP tensors ({n_cb:2d}) : {added_bytes / 1e9:8.3f} GB")
    print(f"[merge]   skipped MTP glue/embed     : {skipped_mtp_bytes / 1e9:8.3f} GB")
    print(f"[merge]   OUTPUT total ({len(out_tensors)} tensors): "
          f"{total / 1e9:8.3f} GB in {n} shards")
    print(f"[merge]   net vs body artifact       : "
          f"{(added_bytes - dropped_bytes) / 1e9:+8.3f} GB")
    print(f"[merge] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
