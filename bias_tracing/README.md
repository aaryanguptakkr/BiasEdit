# Bias Tracing

Trace bias effect in states of language model.

## Tracing
Run the scripts `bash scripts/gpt2m.sh`.

Results are saved in `./results`.

## Histograms
```shell
python fig.py \
    --root results/cross_patch/{direction}/{domain}/causal_trace/cases \
    --bias gender
```

`fig.py` reads the source model, target model, layer count, signed BLANK score,
and per-case NIE baselines from each `.npz` file. Results produced with the old
whole-sentence score must be regenerated before plotting.


Thanks for the original code from [*ROME*](https://github.com/kmeng01/rome).
