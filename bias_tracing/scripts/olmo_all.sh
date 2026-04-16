#!/bin/bash
MODEL="allenai/OLMo-2-0425-1B"
echo "=== [1/4] gender ==="
python experiments/bias_trace.py --model_name=$MODEL --bias_file=data/domain/gender.json
echo "=== [2/4] profession ==="
python experiments/bias_trace.py --model_name=$MODEL --bias_file=data/domain/profession.json
echo "=== [3/4] race ==="
python experiments/bias_trace.py --model_name=$MODEL --bias_file=data/domain/race.json
echo "=== [4/4] religion ==="
python experiments/bias_trace.py --model_name=$MODEL --bias_file=data/domain/religion.json
echo "=== ALL DONE ==="
