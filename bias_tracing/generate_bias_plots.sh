#!/bin/bash
# generate_bias_plots.sh
#
# Runs the full bias-tracing plot pipeline for all models and checkpoints.
#
# Outputs go to:
#   plots/{model}/                        — heatmap, cross-checkpoint figs, stats.json, report.md
#   plots/{model}/{checkpoint_label}/     — per-checkpoint bar charts + composites
#
# Usage:
#   bash generate_bias_plots.sh                         # all models, all domains (zip, ~1 min)
#   bash generate_bias_plots.sh --model pythia-1b       # single model
#   bash generate_bias_plots.sh --bias gender           # single domain
#   bash generate_bias_plots.sh --source local          # read from NFS files (~27 min)
#   bash generate_bias_plots.sh --source auto           # local if extracted, else zip

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "  Bias Tracing Plot Pipeline"
echo "  $(date)"
echo "=================================================="
echo ""

# Step 1: checkpoint × layer heatmaps (fast — reads only peak scores)
echo ">>> Step 1/2: Checkpoint × layer heatmaps"
python scripts/plot_checkpoint_heatmap.py "$@"
echo ""

# Step 2: bar charts, composites, cross-checkpoint grids, stats, reports
echo ">>> Step 2/3: Bar charts, composites, reports"
python fig.py "$@"
echo ""

# Step 3: comparison plots (reads from stats.json — no zip needed)
echo ">>> Step 3/3: Comparison plots (base vs instruct, OLMo vs Pythia, trajectory)"
python scripts/regenerate_compare_plots.py
echo ""

echo "=================================================="
echo "  Done. Results in: plots/"
echo "=================================================="
