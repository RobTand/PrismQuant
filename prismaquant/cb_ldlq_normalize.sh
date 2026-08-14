#!/usr/bin/env bash
# Shared CB LDLQ normalization — sourced by run-pipeline.sh and tests.
# MUST be safe to source: no side effects, no exit, no set -e leakage.
# Usage: source this file, then call normalize_cb_ldlq_vars.
# When sourced with --help, it prints the truth table contract.
# This file is the single source of truth for PRISMAQUANT_CB_LDLQ scope/legacy normalization.

normalize_cb_ldlq_vars() {
  # Normalize PRISMAQUANT_CB_LDLQ and PRISMAQUANT_CB_LDLQ_SCOPE to their canonical
  # forms. Empty or whitespace-only legacy and scope are treated as unset
  # (empty/whitespace -> unset). Padded nonempty values are trimmed and lowercased
  # identically in both Python and shell (e.g. "  NVFP4 " -> "nvfp4", " 1 " -> "1").
  # Truth table (scope authoritative):
  #   neither set -> 0 none
  #   legacy 0, no scope -> 0 none
  #   legacy 1, no scope -> 1 all
  #   scope nvfp4, no legacy -> 1 nvfp4
  #   scope none, no legacy -> 0 none
  #   both set -> legacy must equal (scope != none)
  # Returns 0 on success, 2 on inconsistent explicit pair, and prints to stderr.
  local _was_legacy=0 _was_scope=0 _trimmed

  if [[ -n "${PRISMAQUANT_CB_LDLQ+x}" ]]; then
    _trimmed="$(printf '%s' "${PRISMAQUANT_CB_LDLQ}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ -z "${_trimmed}" ]]; then
      unset PRISMAQUANT_CB_LDLQ
    else
      PRISMAQUANT_CB_LDLQ="$(printf '%s' "${_trimmed}" | tr '[:upper:]' '[:lower:]')"
      _was_legacy=1
    fi
    unset _trimmed
  fi
  if [[ -n "${PRISMAQUANT_CB_LDLQ_SCOPE+x}" ]]; then
    _trimmed="$(printf '%s' "${PRISMAQUANT_CB_LDLQ_SCOPE}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
    if [[ -z "${_trimmed}" ]]; then
      unset PRISMAQUANT_CB_LDLQ_SCOPE
    else
      PRISMAQUANT_CB_LDLQ_SCOPE="$(printf '%s' "${_trimmed}" | tr '[:upper:]' '[:lower:]')"
      _was_scope=1
    fi
    unset _trimmed
  fi

  if (( _was_legacy )); then
    case "${PRISMAQUANT_CB_LDLQ}" in
      1|true|yes|on) PRISMAQUANT_CB_LDLQ=1 ;;
      0|false|no|off) PRISMAQUANT_CB_LDLQ=0 ;;
      *) echo "[normalize] ERROR: PRISMAQUANT_CB_LDLQ must be 0 or 1" >&2; return 2 ;;
    esac
  fi
  if (( _was_scope )); then
    case "${PRISMAQUANT_CB_LDLQ_SCOPE}" in
      none|nvfp4|all) ;;
      *) echo "[normalize] ERROR: PRISMAQUANT_CB_LDLQ_SCOPE must be one of none/nvfp4/all, got ${PRISMAQUANT_CB_LDLQ_SCOPE}" >&2; return 2 ;;
    esac
  fi

  if (( _was_legacy == 0 && _was_scope == 0 )); then
    PRISMAQUANT_CB_LDLQ=0
    PRISMAQUANT_CB_LDLQ_SCOPE="none"
  elif (( _was_legacy == 1 && _was_scope == 0 )); then
    if [[ "${PRISMAQUANT_CB_LDLQ}" == "1" ]]; then
      PRISMAQUANT_CB_LDLQ_SCOPE="all"
    else
      PRISMAQUANT_CB_LDLQ_SCOPE="none"
    fi
  elif (( _was_legacy == 0 && _was_scope == 1 )); then
    if [[ "${PRISMAQUANT_CB_LDLQ_SCOPE}" == "none" ]]; then
      PRISMAQUANT_CB_LDLQ=0
    else
      PRISMAQUANT_CB_LDLQ=1
    fi
  else
    local _expected=0
    if [[ "${PRISMAQUANT_CB_LDLQ_SCOPE}" != "none" ]]; then _expected=1; fi
    if [[ "${PRISMAQUANT_CB_LDLQ}" != "${_expected}" ]]; then
      echo "[normalize] ERROR: PRISMAQUANT_CB_LDLQ=${PRISMAQUANT_CB_LDLQ} inconsistent with PRISMAQUANT_CB_LDLQ_SCOPE=${PRISMAQUANT_CB_LDLQ_SCOPE}" >&2
      return 2
    fi
  fi
  return 0
}

# When executed directly with --normalize, normalize current env and print result.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  if [[ "${1:-}" == "--normalize" ]]; then
    normalize_cb_ldlq_vars || exit $?
    printf '%s %s\n' "${PRISMAQUANT_CB_LDLQ}" "${PRISMAQUANT_CB_LDLQ_SCOPE}"
    exit 0
  fi
  if [[ "${1:-}" == "--help" ]]; then
    echo "Usage: source cb_ldlq_normalize.sh; normalize_cb_ldlq_vars"
    exit 0
  fi
fi
