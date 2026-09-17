#!/bin/bash
# Causal-tracing campaign runner — self-contained, runnable from any account on this server.
#
#   usage:  ./run_all_models.sh <gpu_id> [families] [options]
#
#   <gpu_id>     physical GPU to use, e.g. 7   (the only thing that normally needs changing)
#   [families]   comma-separated: olmo,pythia,qwen,llama,gemma     (default: all)
#   options:
#     --mode=within|cross|both    which patching to run         (default: both)
#                                   within = source == target (self runs + checkpoint runs)
#                                   cross  = source != target (base <-> instruct)
#     --kinds=self,cross,ckpt     finer control than --mode, if you need it
#     --domains=gender,race       restrict to domains          (default: gender,profession,race)
#     --local                     use the owner's own checkout/env/cache and results_v2/
#                                   instead of the shared package (owner's account only)
#     --dry-run                   print the exact run list and exit, touching no GPU
#     --list                      show the full matrix and exit
#
#   examples:
#     ./run_all_models.sh 7                              # everything, on card 7
#     ./run_all_models.sh 7 olmo                         # only the OLMo family
#     ./run_all_models.sh 4 qwen,llama                   # two families on card 4
#     ./run_all_models.sh 6 olmo --mode=within           # OLMo, within-model patching only
#     ./run_all_models.sh 6 qwen --mode=cross            # Qwen, base <-> instruct only
#     ./run_all_models.sh 7 pythia --domains=gender      # checkpoint sweep, gender only
#     ./run_all_models.sh 7 all --dry-run                # preview everything
#
# MODELS COVERED — five families. The four base/instruct pairs support both within- and
# cross-model patching; Pythia has no instruct release, so it is checkpoint (within) only:
#     olmo    allenai/OLMo-2-0425-1B          <-> allenai/OLMo-2-0425-1B-Instruct
#     qwen    Qwen/Qwen2.5-1.5B               <-> Qwen/Qwen2.5-1.5B-Instruct
#     llama   meta-llama/Llama-3.2-1B         <-> meta-llama/Llama-3.2-1B-Instruct
#     gemma   google/gemma-3-1b-pt            <-> google/gemma-3-1b-it
#     pythia  EleutherAI/pythia-1b            6 training checkpoints, within-model only
#
# PARALLELISM: one process saturates an A6000 — measured, 3 concurrent processes on one card
# gave 40.4 forwards/s in total versus 42.3 for a single one. So run ONE process per GPU and
# add GPUs for throughput; never start two on the same card. Different families on different
# cards never collide: they write to different output directories.
#     ./run_all_models.sh 4 olmo  &  ./run_all_models.sh 6 pythia  &  ./run_all_models.sh 7 qwen
#
# RESUMING: safe to re-run at any time. Cases already written are detected and skipped, so an
# interrupted campaign continues where it stopped. Nothing is overwritten — the pipeline
# refuses to write over results produced under a different metric or patch direction.
#
# SURVIVING A DISCONNECT: launch it detached, or it dies with your shell —
#     setsid nohup ./run_all_models.sh 7 olmo > /dev/null 2>&1 &
# A run was lost exactly this way: the process was a child of a terminal session and died
# when that session ended. The per-run logs below are written either way.
#
# STOPPING: kill the campaign by PID (`kill <pid>`). Do NOT use `pkill -f run_all_models` —
# the pattern also matches the shell you type it in, which kills your own session.
#
# WHERE THE OUTPUT GOES — everything lands under one results root, chosen with the layout
# (shared package -> <pkg>/results, owner checkout -> <repo>/results_v2):
#     <results>/<model_base>/<run_name>/causal_trace/cases/*.npz   the numbers, one per case
#     <results>/<model_base>/<run_name>/causal_trace/pdfs/*.pdf    per-case heatmaps
#     <results>/checkpoints/<model>/<branch>/<domain>/...          checkpoint runs
#     <results>/logs/<model>_<domain>.log                          one live log per run
# Aggregate (paper) figures are NOT produced here — that is a separate fig.py step, run by
# the project owner afterwards from these .npz files.
#
# HOW LONG: roughly 8-12 seconds per case; a domain is 800-1500 cases, so 2-5 hours per run
# depending on model and domain. The full matrix is 69 runs, i.e. several days on one card —
# which is why splitting families across GPUs is worth it.
#
# WHAT A SHARED PACKAGE CONTAINS, if you are reading this from one:
#     run_all_models.sh   this file          repo/       pipeline code + StereoSet data
#     env/                python 3.10.19, torch 2.5.1+cu121, transformers 5.3.0
#     hf_cache/           model weights, used offline (HF_HUB_OFFLINE=1, never downloads)
#     results/            output, created on first run
#
# ---------------------------------------------------------------------------------------
# NO PREREQUISITES. The shared package under /deepfreeze/share carries its own code, data,
# python environment and model weights, so it depends on nobody's home directory. Verified:
# a real run opened 16,680 files, none of them under any private home directory.
#
# (--local instead uses the owner's own checkout, environment and model cache; that path is
# for the owner's account only.)
# ---------------------------------------------------------------------------------------

set -uo pipefail          # NOT -e: one failing run must not abort the rest of the campaign

# ---- locations (derived, never hardcoded) -------------------------------------------------
# The script locates itself, so no personal paths live in this file (the project keeps those
# in private_names.py, which is gitignored). Two layouts are recognised:
#
#   <pkg>/run_all_models.sh          + <pkg>/repo + <pkg>/env   -> SHARED PACKAGE
#   <repo>/scripts/run_all_models.sh                            -> OWNER'S CHECKOUT
#
# The shared package is what another account runs: it carries its own code, data, python
# environment and model weights, so it depends on nobody's home directory. That matters
# because /deepfreeze/<user> is a per-user sshfs mount with no allow_other — one account
# simply cannot read another's, whatever the file permissions say. /deepfreeze/share (NFS)
# is the only cross-account path on this cluster.
#
# Override the package location with BIAS_PKG=/some/dir if it is ever moved.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG="${BIAS_PKG:-$HERE}"

# --local forces the owner's checkout even when a package is present. Scanned here because
# paths must be resolved before the main argument parser runs.
LOCAL=""
for a in "$@"; do [[ "$a" == "--local" ]] && LOCAL=1; done

if [[ -z "$LOCAL" && -d "$PKG/repo" && -x "$PKG/env/bin/python" ]]; then
  REPO="$PKG/repo"                       # code + data
  PYTHON="$PKG/env/bin/python"           # python 3.10.19 / torch 2.5.1 / transformers 5.3.0
  export HF_HOME="$PKG/hf_cache"         # model weights, used offline
  RESULTS_ROOT="$PKG/results"            # output — readable by everyone
  WHERE="shared package at $PKG"
else
  # Owner's checkout: this file lives in <repo>/scripts/, so the repo is one level up, and
  # the environment and model cache come from the invoking user's own home.
  REPO="$(cd "$HERE/.." && pwd)"
  PYTHON="$HOME/miniconda3/envs/bias_trace_olmo/bin/python"
  export HF_HOME="$HOME/.cache/huggingface"
  RESULTS_ROOT="$REPO/results_v2"
  WHERE="owner checkout at $REPO${LOCAL:+ — forced by --local}"
fi
export HF_HUB_OFFLINE=1          # never reach the network: use only the cached weights
export PYTHONNOUSERSITE=1        # ignore any ~/.local packages of the invoking account
export TOKENIZERS_PARALLELISM=false

LOGDIR="$RESULTS_ROOT/logs"
OUTPUT_TMPL="$RESULTS_ROOT/{model_base}/{model_name}/causal_trace"
ALL_DOMAINS="gender,profession,race"      # religion excluded by project decision

# ---- the campaign matrix ---------------------------------------------------------------
# family|kind|source|target|branch          (target empty for self/ckpt; branch only for ckpt)
MATRIX=(
  "olmo|self|allenai/OLMo-2-0425-1B||"
  "olmo|self|allenai/OLMo-2-0425-1B-Instruct||"
  "olmo|cross|allenai/OLMo-2-0425-1B|allenai/OLMo-2-0425-1B-Instruct|"
  "olmo|cross|allenai/OLMo-2-0425-1B-Instruct|allenai/OLMo-2-0425-1B|"
  "olmo|ckpt|allenai/OLMo-2-0425-1B||stage1-step10000-tokens21B"

  "pythia|ckpt|EleutherAI/pythia-1b||step0"
  "pythia|ckpt|EleutherAI/pythia-1b||step1000"
  "pythia|ckpt|EleutherAI/pythia-1b||step5000"
  "pythia|ckpt|EleutherAI/pythia-1b||step81000"
  "pythia|ckpt|EleutherAI/pythia-1b||step137000"
  "pythia|ckpt|EleutherAI/pythia-1b||step143000"

  "qwen|self|Qwen/Qwen2.5-1.5B||"
  "qwen|self|Qwen/Qwen2.5-1.5B-Instruct||"
  "qwen|cross|Qwen/Qwen2.5-1.5B|Qwen/Qwen2.5-1.5B-Instruct|"
  "qwen|cross|Qwen/Qwen2.5-1.5B-Instruct|Qwen/Qwen2.5-1.5B|"

  "llama|self|meta-llama/Llama-3.2-1B||"
  "llama|self|meta-llama/Llama-3.2-1B-Instruct||"
  "llama|cross|meta-llama/Llama-3.2-1B|meta-llama/Llama-3.2-1B-Instruct|"
  "llama|cross|meta-llama/Llama-3.2-1B-Instruct|meta-llama/Llama-3.2-1B|"

  "gemma|self|google/gemma-3-1b-pt||"
  "gemma|self|google/gemma-3-1b-it||"
  "gemma|cross|google/gemma-3-1b-pt|google/gemma-3-1b-it|"
  "gemma|cross|google/gemma-3-1b-it|google/gemma-3-1b-pt|"
)

# ---- arguments --------------------------------------------------------------------------
usage() { sed -n '2,30p' "$0"; exit "${1:-0}"; }
[[ $# -eq 0 ]] && usage 1
[[ "$1" == "--list" ]] && { printf '%s\n' "${MATRIX[@]}" | column -t -s'|'; exit 0; }
[[ "$1" == "-h" || "$1" == "--help" ]] && usage 0

GPU="$1"; shift
# The first argument is the GPU id. Validate it: without this check any stray flag or typo
# becomes CUDA_VISIBLE_DEVICES and the campaign starts anyway, on whatever device CUDA picks.
[[ "$GPU" =~ ^[0-9]+$ ]] || { echo "first argument must be a GPU id (a number), got '$GPU'" >&2; usage 1; }
FAMILIES="all"; KINDS=""; MODE=""; DOMAINS_CSV="$ALL_DOMAINS"; DRY_RUN=""
[[ $# -gt 0 && "$1" != --* ]] && { FAMILIES="$1"; shift; }
for a in "$@"; do
  case "$a" in
    --mode=*)    MODE="${a#*=}" ;;
    --kinds=*)   KINDS="${a#*=}" ;;
    --domains=*) DOMAINS_CSV="${a#*=}" ;;
    --local)     ;;   # already handled above, before paths were resolved
    --dry-run)   DRY_RUN=1 ;;
    --list)      printf '%s\n' "${MATRIX[@]}" | column -t -s'|'; exit 0 ;;
    -h|--help)   usage 0 ;;
    *) echo "unknown option: $a" >&2; usage 1 ;;
  esac
done

# --mode is the friendly front-end to --kinds. "within" means source == target, which covers
# both the plain self runs and the checkpoint runs; "cross" means source != target.
if [[ -n "$MODE" ]]; then
  [[ -n "$KINDS" ]] && { echo "use --mode or --kinds, not both" >&2; exit 1; }
  case "$MODE" in
    within) KINDS="self,ckpt" ;;
    cross)  KINDS="cross" ;;
    both)   KINDS="all" ;;
    *) echo "unknown --mode '$MODE' (within|cross|both)" >&2; exit 1 ;;
  esac
fi
KINDS="${KINDS:-all}"
# "all" domains means the three in scope — religion is not part of "all" for this project.
[[ "$DOMAINS_CSV" == "all" ]] && DOMAINS_CSV="$ALL_DOMAINS"
[[ "$FAMILIES" == "all" ]] && FAMILIES="olmo,pythia,qwen,llama,gemma"
# Order the CUDA devices the way nvidia-smi does. Without this, CUDA defaults to
# FASTEST_FIRST and reorders the cards by capability, so on this mixed-GPU host
# (2x RTX 2080 Ti 11G + 2x RTX A6000 48G) the A6000s are promoted to indices 0-1 and
# `<gpu_id> 3` silently lands on the 2080 Ti that nvidia-smi calls card 2 — a different,
# possibly busy, and far smaller card than the one asked for. PCI_BUS_ID makes the id
# passed on the command line mean the same card nvidia-smi shows under that index.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
IFS=',' read -ra DOMAINS <<< "$DOMAINS_CSV"

in_csv() { [[ ",$2," == *",$1,"* ]]; }   # is $1 present in the csv list $2 ?

# ---- preflight: fail loudly here rather than three hours into a run ---------------------
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }
[[ -x "$PYTHON" ]]                         || fail "cannot execute $PYTHON"
[[ -r "$REPO/experiments/bias_trace.py" ]] || fail "cannot read $REPO/experiments/bias_trace.py"
[[ -d "$HF_HOME/hub" ]]                    || fail "cannot reach $HF_HOME/hub (model cache)"
# Probe writability by actually writing: on NFS, `[[ -w ]]` uses access(2), which is
# evaluated server-side and can report "no" on a directory that writes fine (observed on
# /deepfreeze/share, mode 777). A real touch is the only reliable test here.
writable() { local t="$1/.wtest.$$"; ( : > "$t" ) 2>/dev/null && { rm -f "$t"; return 0; }; return 1; }
mkdir -p "$RESULTS_ROOT" "$LOGDIR" 2>/dev/null
writable "$REPO"         || fail "cannot write to $REPO (needed for results/run_log.jsonl)"
writable "$RESULTS_ROOT" || fail "cannot write to $RESULTS_ROOT"
writable "$LOGDIR"       || fail "cannot write to $LOGDIR"
cd "$REPO"                                 || fail "cannot cd to $REPO"   # globals.yml is read from CWD
for d in "${DOMAINS[@]}"; do
  # religion is deliberately out of scope for this project: the domain has only 79 unique
  # cases and loses 28-44% of them to the tokenization guard, leaving too few to support a
  # claim. The data file exists, so this is an explicit refusal rather than an accident.
  [[ "$d" == "religion" ]] && fail "religion is excluded from this project by decision — remove it from --domains"
  [[ -r "data/domain/$d.json" ]] || fail "no such domain file: data/domain/$d.json"
done

# ---- build the run list ------------------------------------------------------------------
RUNS=()      # must be a real empty array: `declare -a` alone trips `set -u` on ${#RUNS[@]}
for cfg in "${MATRIX[@]}"; do
  IFS='|' read -r family kind src tgt branch <<< "$cfg"
  in_csv "$family" "$FAMILIES" || continue
  [[ "$KINDS" == "all" ]] || in_csv "$kind" "$KINDS" || continue
  for domain in "${DOMAINS[@]}"; do RUNS+=("$family|$kind|$src|$tgt|$branch|$domain"); done
done

total=${#RUNS[@]}
[[ $total -eq 0 ]] && { echo "nothing to run for families='$FAMILIES' kinds='$KINDS'" >&2; exit 1; }

echo "=== campaign  gpu=$GPU  families=$FAMILIES  kinds=$KINDS  domains=$DOMAINS_CSV  runs=$total ==="
echo "=== host $(hostname)  env/cache: $WHERE ==="
# Name the physical card, so a mis-numbered run is obvious in the first line of the log
# rather than three hours in. CUDA_DEVICE_ORDER=PCI_BUS_ID above makes this query, which is
# always in nvidia-smi's own order, agree with what the run will actually see as device 0.
gpu_name="$(nvidia-smi --id="$GPU" --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
[[ -n "$gpu_name" ]] || gpu_name="UNKNOWN — nvidia-smi could not read card $GPU"
echo "=== gpu $GPU: $gpu_name ==="
echo "=== started $(date) ==="
[[ -n "$DRY_RUN" ]] && echo "(dry run — no GPU work will be done)"

# ---- run ---------------------------------------------------------------------------------
n=0; failed=0; failed_list=()
for run in "${RUNS[@]}"; do
  IFS='|' read -r family kind src tgt branch domain <<< "$run"
  n=$((n + 1))

  tag="${src##*/}"; [[ -n "$tgt" ]] && tag="${tag}_to_${tgt##*/}"; [[ -n "$branch" ]] && tag="${tag}_${branch}"
  log="$LOGDIR/${tag}_${domain}.log"

  # Argument shape differs per kind. Checkpoint runs need the branch inside the output path,
  # because the generated directory name does not encode the revision.
  case "$kind" in
    self)  args=(--model_name="$src" --output_dir="$OUTPUT_TMPL")
           outdir="results_v2/${src##*/}/ns3_r0_${src##*/}_${domain}/causal_trace" ;;
    cross) args=(--model_source="$src" --model_target="$tgt" --output_dir="$OUTPUT_TMPL")
           outdir="results_v2/${src##*/}_to_${tgt##*/}/ns3_r0_${src##*/}_to_${tgt##*/}_${domain}/causal_trace" ;;
    ckpt)  outdir="results_v2/checkpoints/${src##*/}/$branch/$domain/causal_trace"
           args=(--model_source="$src" --model_target="$src" --branch1="$branch" --branch2="$branch"
                 --output_dir="$outdir") ;;
  esac

  # Resume indicator: how much of this run already exists.
  done_cases=0
  [[ -d "$outdir/cases" ]] && done_cases=$(find "$outdir/cases" -name '*.npz' ! -name '*_attn.npz' ! -name '*_mlp.npz' 2>/dev/null | wc -l)
  resume_note=""; [[ $done_cases -gt 0 ]] && resume_note="  (resuming: $done_cases cases already done)"

  echo "[$n/$total] $family $tag $domain$resume_note"
  if [[ -n "$DRY_RUN" ]]; then
    echo "        $PYTHON experiments/bias_trace.py ${args[*]} --bias_file=data/domain/$domain.json"
    echo "        log -> $log"
    continue
  fi

  started=$(date +%s)
  echo "=== START $tag $domain $(date) (already done: $done_cases) ===" >> "$log"
  if "$PYTHON" experiments/bias_trace.py "${args[@]}" \
       --bias_file="data/domain/$domain.json" >> "$log" 2>&1; then
    echo "=== DONE $tag $domain $(date) after $(( ($(date +%s) - started) / 60 )) min ===" >> "$log"
    echo "        done in $(( ($(date +%s) - started) / 60 )) min"
  else
    rc=$?
    echo "=== FAILED (exit $rc) $tag $domain $(date) ===" >> "$log"
    echo "        FAILED (exit $rc) — continuing; see $log" >&2
    failed=$((failed + 1)); failed_list+=("$tag $domain")
  fi
done

echo "=== campaign finished $(date): $n attempted, $failed failed ==="
for f in "${failed_list[@]:-}"; do [[ -n "$f" ]] && echo "    FAILED: $f"; done
[[ $failed -gt 0 ]] && exit 1 || exit 0
