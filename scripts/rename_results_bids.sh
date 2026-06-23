#!/usr/bin/env bash
# Prepend A/B-registry IDs to the medium-flat reorient pipeline run dirs in
# results/rl, keeping the original descriptive name so paths stay
# self-documenting. results/ is gitignored, so this is plain mv (no git mv).
#
#   aNN_ = canonical Policy A (lift/deliver + the un-freeze-A migration series)
#   bNN_ = canonical Policy B (b01-b13 = the published registry; b14+ = the
#          post-B13 reorient runs in chronological order)
#   bx_  = Policy-B exploration that was NOT canonized into the registry
#
# Foundational grasp / DR / multimorph / other-object / pre-split (inhand v2-v5)
# / eval-scratch dirs are deliberately left untouched.
#
# Usage:  bash scripts/rename_results_bids.sh           # dry-run (default)
#         bash scripts/rename_results_bids.sh --apply    # rename + rewrite refs
set -euo pipefail
cd "$(dirname "$0")/.."
RL=results/rl
APPLY=0; [[ "${1:-}" == "--apply" ]] && APPLY=1

# old-dir-name => new-prefix   (new name = "<prefix>_<old>")
MAP=(
  # ---- Policy A (medium-flat lift / deliver + migration) -------------------
  "20260529-1219-screwdriver_medium_flat_short_proximal_stable_v1|a01"  # FROZEN deliverer (model_500)
  "20260604-2205-policyA_unfreezeA_gripw4|a02"
  "20260604-2206-policyA_unfreezeA_gripw8|a03"
  "20260604-2210-policyA_unfreezeA_gripw16|a04"
  "20260605-1558-policyA_unfreezeA_v2_gripw2_slip5|a05"
  "20260605-1608-policyA_unfreezeA_v2_w2_tol15|a06"
  "20260605-1609-policyA_unfreezeA_v2_w2_tol20|a07"                      # Atol20 (canonical migrated A)
  "20260609-1901-branchB_w4_tol15|a08"                                   # branch-B = un-freeze A
  "20260609-1857-branchB_w6_tol20|a09"
  # ---- Policy B registry (FIXED b01-b13, matches webpaper rl.typ) ----------
  "20260601-1033-policyB_v1|b01"
  "20260602-0024-policyB_v2_smooth10x_quick|b02"
  "20260602-1636-policyB_abl_signed|b03"                                 # signed+critic (model_405)
  "20260603-1746-policyB_p2_lateral_only|b04"                            # lateral-only (best standalone)
  "20260603-1745-policyB_p1_handoff_dronly|b05"
  "20260603-1750-policyB_p3_statebank|b06"
  "20260603-2221-policyB_normallift_v2_fromP2|b07"
  "20260603-2347-policyB_normallift_v3_grace|b08"
  "20260604-1018-policyB_normallift_v3b_repro|b09a"
  "20260604-1019-policyB_normallift_v3b_soft|b09b"
  "20260604-1642-policyB_holdonlyws_repro|b10"                           # hold-only ws, hard onset
  "20260604-1644-policyB_holdonlyws_soft|b11"
  "20260604-1840-policyB_handoff_B12_smooth|b12"
  "20260604-1830-policyB_handoff_B13_softcommit|b13"
  # ---- Policy B post-registry (NEW b14+, chronological) --------------------
  "20260608-1738-policyB_adaptToA_bankA_s40|b14"                         # adapt-B skip-lift bank
  "20260609-1113-policyB_onsetInject_bankA_s40|b15"                      # Badapt (static onset)
  "20260609-1305-policyB_onsetInjectFull_bankA_s40|b16a"                 # complete-state onset
  "20260609-1313-policyB_onsetInjectFull_bankA_s40|b16b"                 # (rerun)
  "20260609-1937-inject_velOnly|b17"                                     # ablation
  "20260609-2013-inject_lastactOnly|b18"                                 # ablation
  "20260609-1813-coadapt_B_toAtol20|b19"
  "20260609-1820-B_complete_fromBadapt|b20"
  "20260609-2116-coadapt_B10_Atol20_static|b21a"                        # cross-pairing eval
  "20260609-2121-coadapt_B10_Atol20_static|b21b"
  "20260609-2157-coadapt_Badapt_Atol20_static|b22"                       # co-adapt best (0.0114)
  "20260610-1043-policyB_liveAreset_smoke|b23"
  "20260610-1046-policyB_liveAreset_fromB10|b24"                         # *** live-A WIN (seam held) ***
  "20260610-1202-policyB_liveAreset_fromB10_s05|b25"                     # canonical 0.5 retrain
  "20260610-1348-policyB_liveAreset_fromB4|b26"                          # B4 warmstart (collapsed)
  "20260610-1355-policyB_liveAreset_cont40M|b27"                         # 40M continuation
  "20260611-1147-policyB_liveAreset_B10qual_commitbonus|b28"             # B10 quality + commit bonus (held-cos 0.759)
  "20260611-1152-policyB_liveAreset_B10qual_commit60|b29"                # *best quality so far: commit60 + basin re-anneal, held-cos 0.784*
  # ---- Policy B explorations, NOT canonized (bx_) --------------------------
  "20260601-2310-policyB_v2_smooth5x|bx"
  "20260601-2311-policyB_v2_smooth10x|bx"
  "20260602-0024-policyB_v2_smooth5x_quick|bx"
  "20260602-1323-policyB_v21_decenter_w15.0|bx"
  "20260602-1323-policyB_v21_decenter_w40.0|bx"
  "20260602-1424-policyB_v3_brace8|bx"
  "20260602-1424-policyB_v3_brace20|bx"
  "20260602-1541-policyB_abl_control|bx"
  "20260602-1541-policyB_abl_signed|bx"
  "20260602-1635-policyB_abl_control|bx"
  "20260602-1704-policyB_abl_control_smooth|bx"
  "20260602-1704-policyB_abl_signed_smooth|bx"
  "20260602-1731-policyB_v2_smooth5x|bx"
  "20260602-1732-policyB_v2_smooth10x|bx"
  "20260603-1117-policyB_v2_smooth5x|bx"
  "20260603-1117-policyB_v2_smooth10x|bx"
  "20260603-1208-policyB_rr_lateral|bx"
  "20260603-1208-policyB_rr_quick|bx"
  "20260603-1315-policyB_normallift|bx"
  "20260603-1404-policyB_hdr_hi|bx"
  "20260603-1404-policyB_hdr_lo|bx"
  "20260603-1448-policyB_hdrA_dr_lateral|bx"
  "20260603-1452-policyB_hdrB_curric_lateral|bx"
  "20260603-1538-policyB_hdrA2_curric_lat8|bx"
  "20260603-2349-policyB_normallift_v3_holdonly|bx"                      # hold-only CONTROL
  # B4-warmstart live-A seam attempts — all FAILED (B4 drops A's flat delivery; NaN-prone)
  "20260611-1145-policyB_liveAreset_fromB4_survwin|bx"
  "20260611-1146-policyB_liveAreset_fromB4_easein|bx"
  "20260611-1151-policyB_liveAreset_fromB4_survwin_r2|bx"
  # adapt-B-from-banked-A, take 5-6 — CYCLICAL (B6/B14/B15/b17/b18 already did this); killed, dropped
  "20260611-1444-policyB_adaptB3_fromAbank|bx"
  "20260611-1446-policyB_adaptB4_fromAbank_full|bx"
)

# Optional human descriptions / key results, keyed by ID (others auto-derive from dir name).
declare -A DESC=(
  [a01]="FROZEN Policy A lift+deliver (model_500) — kept frozen"
  [a07]="Atol20 = migrated A (frozen-A → B10 grip)"
  [b03]="signed+critic reorienter (model_405), held-cos 0.979 — SMOOTHER (user pref)"
  [b04]="lateral-only, best STANDALONE reorienter held-cos 0.988"
  [b10]="hold-only warmstart, first to survive delivery but violent"
  [b24]="live-A reset (warmstart B10) — FIRST to HOLD the continuous seam (min-z 0.110, cos 0.751)"
  [b29]="live-A + commit60 + basin re-anneal — BEST handoff policy so far (held-cos 0.784)"
  [bx]="uncanonized exploration / superseded / failed (see dir name)"
)

REF_PATHS=(docs scripts webpaper RESEARCH_STATE.md)
n_ren=0; n_miss=0; n_skip=0
for entry in "${MAP[@]}"; do
  old="${entry%%|*}"; pref="${entry#*|}"; new="${pref}_${old}"
  if [[ ! -d "$RL/$old" ]]; then
    if [[ -d "$RL/$new" ]]; then echo "skip (already): $new"; n_skip=$((n_skip+1));
    else echo "MISSING:       $RL/$old"; n_miss=$((n_miss+1)); fi
    continue
  fi
  echo "$old  ->  $new"
  n_ren=$((n_ren+1))
  if [[ $APPLY -eq 1 ]]; then
    mv "$RL/$old" "$RL/$new"
    # rewrite every reference results/rl/<old>  ->  results/rl/<new>
    # (|| true: grep exits 1 when a dir has no references — must not trip set -e)
    files=$(grep -rl --include='*.md' --include='*.typ' --include='*.sh' --include='*.py' \
        "results/rl/$old" "${REF_PATHS[@]}" 2>/dev/null || true)
    if [[ -n "$files" ]]; then
      while read -r f; do sed -i "s#results/rl/$old#results/rl/$new#g" "$f"; done <<<"$files"
    fi
  fi
done
echo "---- $([[ $APPLY -eq 1 ]] && echo APPLIED || echo DRY-RUN): rename=$n_ren skip=$n_skip missing=$n_miss ----"

# ---- self-healing: any UNPREFIXED policyA/policyB run dir -> bx_ (safe default) ----
# Guarantees no run is ever left in the bare "<ts>-tag" convention. Promote a bx_
# run to a canonical aNN/bNN by adding it to the MAP above and re-running --apply.
# Scoped to *policyA*/*policyB* so foundational/DR/other-object dirs are untouched.
shopt -s nullglob
AUTOBX=()
for d in "$RL"/[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]-[0-9][0-9][0-9][0-9]-*policy[AB]*/; do
  b=$(basename "$d"); new="bx_$b"
  echo "auto-bx (unmatched policy run): $b  ->  $new"
  AUTOBX+=("$new")
  if [[ $APPLY -eq 1 ]]; then
    mv "$RL/$b" "$RL/$new"
    files=$(grep -rl --include='*.md' --include='*.typ' --include='*.sh' --include='*.py' \
        "results/rl/$b" "${REF_PATHS[@]}" 2>/dev/null || true)
    [[ -n "$files" ]] && while read -r f; do sed -i "s#results/rl/$b#results/rl/$new#g" "$f"; done <<<"$files"
  fi
done
[[ ${#AUTOBX[@]} -gt 0 ]] && echo "---- auto-bx'd ${#AUTOBX[@]} unmatched policy run(s) ----"

# ---- emit the meta-registry: single source of truth for A/B policy versions ----
# Always regenerated from the MAP above (so MAP is the ONLY thing to edit). Lists
# every canonical ID -> run dir + description. New runs: add to MAP, re-run --apply.
REG="$RL/REGISTRY.md"
{
  echo "# results/rl — A/B policy registry"
  echo
  echo "**Auto-generated by \`scripts/rename_results_bids.sh\` — do NOT hand-edit.** To add/rename a"
  echo "run: append it to the \`MAP\` in that script (and \`DESC\` if you want a blurb), then run"
  echo "\`bash scripts/rename_results_bids.sh --apply\`. Roles: **aNN**=lift/deliver, **bNN**=reorient"
  echo "(b01–b13 = the published registry; b14+ chronological), **bx_**=uncanonized/superseded/failed."
  echo
  echo "| ID | run dir | description |"
  echo "|----|---------|-------------|"
  for entry in "${MAP[@]}"; do
    old="${entry%%|*}"; pref="${entry#*|}"; new="${pref}_${old}"
    d="${DESC[$pref]:-}"
    [[ -z "$d" ]] && d=$(echo "$old" | sed -E 's/^[0-9]{8}-[0-9]{4}-//; s/^policy[AB]_//; s/_/ /g')
    mark=""; [[ -d "$RL/$new" ]] || mark=" ⚠missing"
    echo "| \`$pref\` | \`$new\`$mark | $d |"
  done
  for new in "${AUTOBX[@]}"; do
    desc=$(echo "$new" | sed -E 's/^bx_[0-9]{8}-[0-9]{4}-//; s/^policy[AB]_//; s/_/ /g')
    echo "| \`bx\` | \`$new\` | $desc (auto-bx — promote via MAP if canonical) |"
  done
} > "$REG"
echo "---- wrote meta-registry: $REG ($(grep -c '^| ' "$REG") rows) ----"
