import argparse
import json
import os, sys
import time, datetime
# os.chdir(sys.path[0])
sys.path.append("./")
import re
from collections import defaultdict

import numpy
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    LlamaTokenizer, 
    LlamaForCausalLM,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from dsets import StereoSetDataset
from util import nethook
from tqdm import tqdm

SCORE_METRIC = "blank_logprob_stereo_minus_anti_v1"
SCORE_METRIC_SENTENCE = "sentence_logprob_stereo_minus_anti_v1"   # |value| == legacy whole-sentence abs metric


def sentence_labels(inp, pad_id):
    """Whole-sentence labels (legacy scoring span): every real token except position 0."""
    ids = inp["input_ids"]
    labels = ids.clone() if pad_id is None else torch.where(ids != pad_id, ids, torch.full_like(ids, -100))
    labels[:, 0] = -100     # start marker / BOS
    return labels


def main():
    parser = argparse.ArgumentParser(description="Causal Tracing")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    def parse_noise_rule(code):
        if code in ["m", "s"]:
            return code
        elif re.match(r"^[uts][\d\.]+", code):
            return code
        else:
            return float(code)

    aa(
        "--model_source",
        default="allenai/OLMo-2-0425-1B",
        choices=[
            "llama-2-7b",
            "gpt2-xl",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
            "gpt-j-6b",
            "allenai/OLMo-2-0425-1B",
            "allenai/OLMo-2-0425-1B-Instruct",
            "EleutherAI/pythia-1b",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "google/gemma-3-1b-pt",
            "google/gemma-3-1b-it",
        ],
    )
    aa(
        "--model_target",
        default="allenai/OLMo-2-0425-1B-Instruct",
        choices=[
            "llama-2-7b",
            "gpt2-xl",
            "gpt2-large",
            "gpt2-medium",
            "gpt2",
            "gpt-j-6b",
            "allenai/OLMo-2-0425-1B",
            "allenai/OLMo-2-0425-1B-Instruct",
            "EleutherAI/pythia-1b",
            "Qwen/Qwen2.5-1.5B",
            "Qwen/Qwen2.5-1.5B-Instruct",
            "meta-llama/Llama-3.2-1B",
            "meta-llama/Llama-3.2-1B-Instruct",
            "google/gemma-3-1b-pt",
            "google/gemma-3-1b-it",
        ],
    )
    aa("--branch1", default=None)
    aa("--branch2", default=None)
    aa("--bias_file", default="data/domain/gender.json")
    aa("--subject_file", default="data/knowns.json")          # set(stereoset target + stereoset subject + jieyu)
    aa("--output_dir", default="results/{model_base}/{model_name}/causal_trace")
    aa("--noise_level", default="s3", type=parse_noise_rule)
    aa("--replace", default=0, type=int)
    aa("--pattern", default="all", choices=["all", "one"], type=str)
    aa("--samples", default=10, type=int)
    args = parser.parse_args()

    model_base_source = args.model_source.split("/")[-1]
    model_base_target = args.model_target.split("/")[-1]
    model_base = f"{model_base_source}_to_{model_base_target}"
    modeldir = f'r{args.replace}_{model_base.replace("/", "_")}'
    modeldir = f"n{args.noise_level}_" + modeldir + f"_{args.bias_file.split('/')[-1].split('.')[0]}"
    output_dir = args.output_dir.format(model_name=modeldir, model_base=model_base)
    result_dir = f"{output_dir}/cases"
    pdf_dir = f"{output_dir}/pdfs"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    def get_dtype(model_name):
        # Use corresponding model config from huggingface config
        if 'olmo' in model_name.lower():
            if 'instruct' in model_name.lower():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float32
        elif 'pythia' in model_name.lower():
            torch_dtype = torch.float16
        elif any(k in model_name.lower() for k in ("qwen", "llama-3", "gemma")):
            torch_dtype = torch.bfloat16
        else:
            raise ValueError(f"No dtype rule for {model_name}")
        return torch_dtype

    torch_dtype_source = get_dtype(args.model_source)
    torch_dtype_target = get_dtype(args.model_target)

    mt_source = ModelAndTokenizer(args.model_source, torch_dtype=torch_dtype_source, branch=args.branch1)
    mt_target = ModelAndTokenizer(args.model_target, torch_dtype=torch_dtype_target, branch=args.branch2)

    # Embedding
    subjects = json.load(open(args.subject_file))

    # Bias Dataset
    knowns = StereoSetDataset(mt_target.tokenizer, args.bias_file, args.model_target)

    # TODO: check
    # when patching mt1 clean -> mt2 corrupted do you use noise from mt2?
    noise_level = args.noise_level
    uniform_noise = False
    if isinstance(noise_level, str):
        if noise_level.startswith("s"):
            # Automatic spherical gaussian
            factor = float(noise_level[1:]) if len(noise_level) > 1 else 1.0
            noise_level = factor * collect_embedding_std(
                mt_target, subjects
            )
            print(f"Using noise_level {noise_level} to match model times {factor}")
        elif noise_level.startswith("u"):
            uniform_noise = True
            noise_level = float(noise_level[1:])

    run_start = time.time()
    run_log_path = "results/run_log.jsonl"
    os.makedirs("results", exist_ok=True)
    domain = args.bias_file.split("/")[-1].split(".")[0]
    start_entry = {
        "status": "started",
        "model_name": model_base,
        "domain": domain,
        "output_dir": output_dir,
        "num_layers": mt_target.num_layers,
        "num_samples": len(knowns),
        "start_time": datetime.datetime.now().isoformat(),
    }
    with open(run_log_path, "a") as _log_f:
        _log_f.write(json.dumps(start_entry) + "\n")
    print(f"[run_log] started: model={model_base} domain={domain} layers={mt_target.num_layers} samples={len(knowns)}")

    for knowledge in tqdm(knowns):
        known_id = knowledge["id"]

        # Clean target-model preference at the BLANK fill.
        inp_anti, e_range_anti, blank_idxs_anti, inp_anti_origin = make_inputs(mt_target, prompts=[knowledge['anti']] * (args.samples + 1), labels=[knowledge['anti_mask']] * (args.samples + 1), subject=knowledge['subject'], blank_idxs=knowledge.get('anti_blank_idxs'))
        inp_stereo, e_range_stereo, blank_idxs_stereo, inp_stereo_origin = make_inputs(mt_target, prompts=[knowledge['stereo']] * (args.samples + 1), labels=[knowledge['stereo_mask']] * (args.samples + 1), subject=knowledge['subject'], blank_idxs=knowledge.get('stereo_blank_idxs'))
        if inp_anti is None or inp_stereo is None:
            print(f"Skipping {known_id}: subject or BLANK span could not be aligned")
            continue
        if inp_anti["input_ids"].shape[1] != inp_stereo["input_ids"].shape[1]:
            print(f"Skipping {known_id}: anti/stereo token lengths differ")
            continue
        sl_anti = sentence_labels(inp_anti, mt_target.tokenizer.pad_token_id)
        sl_stereo = sentence_labels(inp_stereo, mt_target.tokenizer.pad_token_id)
        with torch.no_grad():
            pred_anti = _logits(mt_target.model(**inp_anti))
            targ_anti = inp_anti["labels"]
            pred_stereo = _logits(mt_target.model(**inp_stereo))
            targ_stereo = inp_stereo["labels"]
            base_score = causal_difference(pred_anti, targ_anti, pred_stereo, targ_stereo)
            base_score_sentence = causal_difference(pred_anti, sl_anti, pred_stereo, sl_stereo)
            print(base_score, base_score_sentence)
        
        # Preference after corrupting the subject embeddings.
        anti_outputs = trace_with_patch(
            model_source=mt_source.model,
            model_target=mt_target.model,
            inp=inp_anti, 
            states_to_patch=[], 
            tokens_to_mixs=e_range_anti, # bias attribute words
            noise=noise_level, 
            uniform_noise=uniform_noise
        )

        stereo_outputs = trace_with_patch(
            model_source=mt_source.model,
            model_target=mt_target.model,
            inp=inp_stereo, 
            states_to_patch=[], 
            tokens_to_mixs=e_range_stereo, 
            noise=noise_level, 
            uniform_noise=uniform_noise
        )
        # Outputs with corrupted the embedding of bias attribute words
        pred_anti = _logits(anti_outputs)
        targ_anti = inp_anti["labels"]
        pred_stereo = _logits(stereo_outputs)
        targ_stereo = inp_stereo["labels"]

        # Corrupted target-model preference, corrupted rows only (both scoring spans).
        low_score = causal_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
        low_score_sentence = causal_difference(pred_anti[1:], sl_anti[1:], pred_stereo[1:], sl_stereo[1:])

        for kind in None, "mlp", "attn":
            print(f"Causal Tracing for {known_id} {kind} ==========================================================")
            kind_suffix = f"_{kind}" if kind else ""
            filename = f"{result_dir}/knowledge_{known_id}{kind_suffix}.npz"
            numpy_result = None
            if os.path.isfile(filename):
                with numpy.load(filename, allow_pickle=True) as cached:
                    cached_result = dict(cached)
                cached_metric = str(numpy.asarray(cached_result.get("score_metric", "")).item())
                cached_source = str(numpy.asarray(cached_result.get("source_model", "")).item())
                cached_target = str(numpy.asarray(cached_result.get("target_model", "")).item())
                if (cached_metric == SCORE_METRIC and
                        cached_source == args.model_source and
                        cached_target == args.model_target):
                    numpy_result = cached_result
                else:
                    raise RuntimeError(
                        f"Refusing to overwrite {filename}: it was produced under a different "
                        f"metric/direction (metric={cached_metric or 'legacy'}). "
                        f"Old results are never overwritten — use a fresh --output_dir.")

            if numpy_result is None:
                result = calculate_hidden_flow(
                    mt_source,
                    mt_target,
                    knowledge,
                    inp_anti,                       # context
                    inp_stereo,
                    e_range_anti,                   # bias attribute word
                    e_range_stereo,
                    blank_idxs_anti,                # bias term
                    blank_idxs_stereo,
                    inp_anti_origin,                # label for context
                    inp_stereo_origin,                    
                    noise=noise_level,
                    uniform_noise=uniform_noise,
                    replace=args.replace,
                    kind=kind,
                )
                if not result:
                    print(f"Skipping {knowledge['id']}")
                    continue
                result["high_score"] = base_score
                result["low_score"] = low_score
                result["nie"] = normalized_indirect_effect(result["scores"], base_score, low_score)
                result["high_score_sentence"] = base_score_sentence
                result["low_score_sentence"] = low_score_sentence
                result["nie_sentence"] = normalized_indirect_effect(
                    result["scores_sentence"], base_score_sentence, low_score_sentence)
                numpy_result = {
                    # MODIFIED: to prevent unsupported ScalarType BFloat16 in numpy cast to float32 first
                    k: v.detach().to(torch.float32).cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in result.items()
                }
                numpy.savez(filename, **numpy_result)
            # if not numpy_result["correct_prediction"]:
            #     tqdm.write(f"Skipping {knowledge['prompt']}")
            #     continue
            plot_result = dict(numpy_result)
            plot_result["kind"] = kind
            pdfname = f'{pdf_dir}/{known_id}_{str("_".join(numpy_result["subject"])).strip()}_{kind_suffix}'
            plot_trace_heatmap(plot_result, savepdf_pre=pdfname, modelname=mt_target.model_name)


def trace_with_patch(
    model_source, # The source model to patch from
    model_target,  # The target model to patch to
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    tokens_to_mixs, # Range List of tokens to corrupt (begin, end); subject tokens range
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mixs specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """

    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model_target, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x
    
    # =========================================================================
    # MODIFIED: Extract Clean States from the Source Model
    # We must run this completely before defining the patch rule so the 
    # data actually exists when the target model goes looking for it.
    # =========================================================================
    layers_to_trace = list(patch_spec.keys())
    with torch.no_grad(), nethook.TraceDict(model_source, layers_to_trace) as source_trace:
        model_source(**inp)

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            # corrupt all subjects
            if tokens_to_mixs is not None:
                for tokens_to_mix in tokens_to_mixs:
                    b, e = tokens_to_mix
                    noise_data = noise_fn(
                        torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                    ).to(x.device)
                    if replace:
                        x[1:, b:e] = noise_data
                    else:
                        x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        
        h = untuple(x)

        # MODIFIED: Reach back into cache and grab the source model's clean state
        source_clean_h = untuple(source_trace[layer].output)

        for t in patch_spec[layer]:
            h[1:, t] = source_clean_h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers

    with torch.no_grad(), nethook.TraceDict(
        model_target,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model_target(**inp)

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return outputs_exp, all_traced

    return outputs_exp

def trace_with_repatch(
    model,  # The model
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    states_to_unpatch,  # A list of (token index, layername) triples to re-randomize
    answers_t,  # Answer probabilities to collect
    tokens_to_mixs,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
):
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    unpatch_spec = defaultdict(list)
    for t, l in states_to_unpatch:
        unpatch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mixs is not None:
                for tokens_to_mix in tokens_to_mixs:
                    b, e = tokens_to_mix
                    x[1:, b:e] += noise * torch.from_numpy(
                        prng(x.shape[0] - 1, e - b, x.shape[2])
                    ).to(x.device)
            return x
        if first_pass or (layer not in patch_spec and layer not in unpatch_spec):
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec.get(layer, []):
            h[1:, t] = h[0, t]
        for t in unpatch_spec.get(layer, []):
            h[1:, t] = untuple(first_pass_trace[layer].output)[1:, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    for first_pass in [True, False] if states_to_unpatch else [False]:
        with torch.no_grad(), nethook.TraceDict(
            model,
            [embed_layername] + list(patch_spec.keys()) + list(unpatch_spec.keys()),
            edit_output=patch_rep,
        ) as td:
            outputs_exp = model(**inp)
            if first_pass:
                first_pass_trace = td

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    return probs


def _logits(x):
    return x if not hasattr(x, "logits") else x.logits

def causal_difference(pred_anti, targ_anti, pred_stereo, targ_stereo):
    if pred_anti.dim() == 3 and pred_stereo.dim() == 3:  # Dealing with sequences
        pred_anti = pred_anti[:, :-1]  # Remove last prediction in sequence
        targ_anti = targ_anti[:, 1:]  # Shift to align predictions and targets
        pred_stereo = pred_stereo[:, :-1]
        targ_stereo = targ_stereo[:, 1:]

    def get_score(predictions, targets):
        mask = targets != -100
        if not mask.any():
            raise ValueError("No scoreable BLANK tokens remain after causal-LM shifting")
        token_log_probs = predictions.float().log_softmax(-1)[mask]
        target_ids = targets[mask].long().unsqueeze(1)
        return token_log_probs.gather(1, target_ids).mean()

    anti_score = torch.stack([get_score(p, t) for p, t in zip(pred_anti, targ_anti)])
    stereo_score = torch.stack([get_score(p, t) for p, t in zip(pred_stereo, targ_stereo)])
    return stereo_score.mean() - anti_score.mean()


def normalized_indirect_effect(scores, high_score, low_score):
    high_score = torch.as_tensor(high_score, device=scores.device, dtype=scores.dtype)
    low_score = torch.as_tensor(low_score, device=scores.device, dtype=scores.dtype)
    gap = high_score - low_score
    if torch.abs(gap).item() < 1e-8:
        return torch.full_like(scores, float("nan"))
    return (scores - low_score) / gap

def calculate_hidden_flow(
    mt_source,
    mt_target,
    knowledge,
    inp_anti, 
    inp_stereo,
    e_range_anti, 
    e_range_stereo,
    blank_idxs_anti, 
    blank_idxs_stereo,
    inp_anti_origin,
    inp_stereo_origin,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    window=10,
    kind=None,
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    
    
    # difference after corrupting embedding and restoring
    
    if not kind:
        differences, differences_sentence = trace_important_states(
            mt_source,
            mt_target,
            inp_anti, inp_stereo,
            e_range_anti, e_range_stereo,
            blank_idxs_anti, blank_idxs_stereo,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            # token_range=token_range,
        )
    else:
        differences, differences_sentence = trace_important_window(
            mt_source,
            mt_target,
            inp_anti, inp_stereo,
            e_range_anti, e_range_stereo,
            blank_idxs_anti, blank_idxs_stereo,
            noise=noise,
            uniform_noise=uniform_noise,
            replace=replace,
            window=window,
            kind=kind,
            # token_range=token_range,
        )
    differences = differences.detach().cpu()                            #(seq_len, num_layers)
    differences_sentence = differences_sentence.detach().cpu()
    return dict(
        scores=differences,
        scores_sentence=differences_sentence,
        score_metric=SCORE_METRIC,
        score_metric_sentence=SCORE_METRIC_SENTENCE,
        case_id=knowledge["id"],
        source_model=mt_source.model_name,
        target_model=mt_target.model_name,
        direction=f"{mt_source.model_name} -> {mt_target.model_name}",
        num_layers=mt_target.num_layers,
        anti_input_ids=inp_anti["input_ids"][0],                               # input_ids of the prompt
        stereo_input_ids=inp_stereo['input_ids'][0],
        input_tokens_anti=decode_tokens(mt_target.tokenizer, inp_anti["input_ids"][0]),      # tokens of the prompt
        input_tokens_stereo=decode_tokens(mt_target.tokenizer, inp_stereo["input_ids"][0]),
        corrupt_range_anti=e_range_anti,
        corrupt_range_stereo=e_range_stereo,
        blank_idxs_anti=blank_idxs_anti,
        blank_idxs_stereo=blank_idxs_stereo,           # bias term idx range in input_ids
        subject=knowledge['subject'],                  
        window=window,
        # correct_prediction=True,
        kind=kind or "",
    )


def trace_important_states(
    mt_source,
    mt_target,
    inp_anti, inp_stereo,
    e_range_anti, e_range_stereo,
    blank_idxs_anti, blank_idxs_stereo,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    # token_range=None,
):
    ntoks_anti = inp_anti["input_ids"].shape[1]
    ntoks_stereo = inp_stereo['input_ids'].shape[1]
    assert ntoks_anti == ntoks_stereo

    
    # if token_range is None:
    # token_range_anti = list(set(range(ntoks_anti)) - set(blank_idxs_anti))
    # token_range_stereo = list(set(range(ntoks_stereo)) - set(blank_idxs_stereo))
    # assert len(token_range_stereo) == len(token_range_anti), "After remove blank tokens, anti and stereo should have the same length"
    
    sl_anti = sentence_labels(inp_anti, mt_target.tokenizer.pad_token_id)
    sl_stereo = sentence_labels(inp_stereo, mt_target.tokenizer.pad_token_id)
    table = [] # (num_layers, seq_len)
    table_sentence = []

    for tnum in range(ntoks_anti):
        row = []
        row_sentence = []
        for layer in range(mt_target.num_layers):
            anti_outputs = trace_with_patch(
                model_source=mt_source.model,
                model_target=mt_target.model,
                inp=inp_anti,
                states_to_patch=[(tnum, layername(mt_target.model, layer))],
                tokens_to_mixs=e_range_anti,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            stereo_outputs = trace_with_patch(
                model_source=mt_source.model,
                model_target=mt_target.model,
                inp=inp_stereo,
                states_to_patch=[(tnum, layername(mt_target.model, layer))],
                tokens_to_mixs=e_range_stereo,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )
            pred_anti = _logits(anti_outputs)
            targ_anti = inp_anti["labels"]
            pred_stereo = _logits(stereo_outputs)
            targ_stereo = inp_stereo["labels"]

            # log-prob gap on the corrupted rows, both scoring spans (same logits)
            r = causal_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
            r_sent = causal_difference(pred_anti[1:], sl_anti[1:], pred_stereo[1:], sl_stereo[1:])
            row.append(r)
            row_sentence.append(r_sent)
        table.append(torch.stack(row))
        table_sentence.append(torch.stack(row_sentence))
    return torch.stack(table), torch.stack(table_sentence)


def trace_important_window(
    mt_source,
    mt_target,
    inp_anti, inp_stereo,
    e_range_anti, e_range_stereo,
    blank_idxs_anti, blank_idxs_stereo,
    kind,
    window=10,
    noise=0.1,
    uniform_noise=False,
    replace=False,
    # token_range=None,
):
    ntoks_anti = inp_anti["input_ids"].shape[1]
    ntoks_stereo = inp_stereo['input_ids'].shape[1]
    assert ntoks_anti == ntoks_stereo

     # if token_range is None:
    # token_range_anti = list(set(range(ntoks_anti)) - set(blank_idxs_anti))
    # token_range_stereo = list(set(range(ntoks_stereo)) - set(blank_idxs_stereo))
    # assert len(token_range_stereo) == len(token_range_anti), "After remove blank tokens, anti and stereo should have the same length"
    
    sl_anti = sentence_labels(inp_anti, mt_target.tokenizer.pad_token_id)
    sl_stereo = sentence_labels(inp_stereo, mt_target.tokenizer.pad_token_id)
    table = [] # (num_layers, seq_len)
    table_sentence = []
    for tnum in range(ntoks_anti):
        row = []
        row_sentence = []
        for layer in range(mt_target.num_layers):
            layerlist_anti = [
                (tnum, layername(mt_target.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(mt_target.num_layers, layer - (-window // 2))
                )
            ]
            anti_outputs = trace_with_patch(
                mt_source.model,
                mt_target.model,
                inp=inp_anti,
                states_to_patch=layerlist_anti,
                tokens_to_mixs=e_range_anti,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )

            layerlist_stereo = [
                (tnum, layername(mt_target.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(mt_target.num_layers, layer - (-window // 2))
                )
            ] 
            stereo_outputs = trace_with_patch(
                model_source=mt_source.model,
                model_target=mt_target.model,
                inp=inp_stereo,
                states_to_patch=layerlist_stereo,
                tokens_to_mixs=e_range_stereo,
                noise=noise,
                uniform_noise=uniform_noise,
                replace=replace,
            )

            pred_anti = _logits(anti_outputs)
            targ_anti = inp_anti["labels"]
            pred_stereo = _logits(stereo_outputs)
            targ_stereo = inp_stereo["labels"]
            # log-prob gap on the corrupted rows, both scoring spans (same logits)
            r = causal_difference(pred_anti[1:], targ_anti[1:], pred_stereo[1:], targ_stereo[1:])
            r_sent = causal_difference(pred_anti[1:], sl_anti[1:], pred_stereo[1:], sl_stereo[1:])
            row.append(r)
            row_sentence.append(r_sent)
        table.append(torch.stack(row))
        table_sentence.append(torch.stack(row_sentence))
    return torch.stack(table), torch.stack(table_sentence)


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        branch=None
    ):
        if tokenizer is None:
            assert model_name is not None
            if any(k in model_name.lower() for k in ("qwen", "llama-3", "gemma")):
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            elif "llama" in model_name.lower():
                tokenizer = LlamaTokenizer.from_pretrained(model_name)
            elif model_name=="gpt2-medium":
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                
        if model is None:
            assert model_name is not None
            if any(k in model_name.lower() for k in ("qwen", "llama-3", "gemma")):
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype,
                    revision=branch,
                )
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token   # llama-3.2 only; qwen/gemma ship pad tokens
            elif "llama" in model_name.lower():
                model = LlamaForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                model.model.embed_tokens.weight.data[-1] = model.model.embed_tokens.weight.data.mean(0)
            elif "gpt" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype
                )
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                model.resize_token_embeddings(len(tokenizer))
                model.transformer.wte.weight.data[-1] = model.transformer.wte.weight.data.mean(0)
            elif "olmo" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype,
                    revision=branch,
                    trust_remote_code=True
                )
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
                    model.model.embed_tokens.weight.data[-1] = model.model.embed_tokens.weight.data.mean(0)
            elif "pythia" in model_name.lower():
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype,
                    revision=branch
                )
                if tokenizer.pad_token is None:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
                    model.gpt_neox.embed_in.weight.data[-1] = model.gpt_neox.embed_in.weight.data.mean(0)
            else:
                raise ValueError(f"Unsupported model {model_name}: this pipeline is causal-only")
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [
            n
            for n, m in model.named_modules()
            if (re.match(r"^(transformer|gpt_neox|model)\.(h|layers)\.\d+$", n))
        ]
        self.num_layers = len(self.layer_names)
        self.iscausal = True   # causal-only pipeline
        self.model_name = model_name

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def layername(model, num, kind=None):
    if hasattr(model, "transformer"):
        if kind == "embed":
            return "transformer.wte"
        return f'transformer.h.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "gpt_neox"):
        if kind == "embed":
            return "gpt_neox.embed_in"
        if kind == "attn":
            kind = "attention"
        return f'gpt_neox.layers.{num}{"" if kind is None else "." + kind}'
    if hasattr(model, "model"): # llama
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            return f'model.layers.{num}.self_attn'
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"



def plot_trace_heatmap(result, savepdf_pre=None, title=None, xlabel=None, modelname=None):
    differences = result["nie"]
    color_limit = numpy.nanmax(numpy.abs(differences))
    if not numpy.isfinite(color_limit) or color_limit == 0:
        color_limit = 1.0
    answer = result["subject"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels_anti = list(result["input_tokens_anti"])
    labels_stereo = list(result['input_tokens_stereo'])

    # anti
    for e_range in result['corrupt_range_anti']:
        for i in range(*e_range):
            labels_anti[i] = labels_anti[i] + "*"
    labels_anti[result['blank_idxs_anti'][0]] = "[" + labels_anti[result['blank_idxs_anti'][0]].strip()
    labels_anti[result['blank_idxs_anti'][1]-1] = labels_anti[result['blank_idxs_anti'][1]-1].strip() + "]"

    savepdf = savepdf_pre+"_anti.pdf"
    with plt.rc_context(rc={"font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1], 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1], 5)))
        ax.set_yticklabels(labels_anti)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after the corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        cb.set_label("Normalized indirect effect (NIE)")
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
            
            cb.ax.set_title(f"Corrupt: {' '.join(answer)}", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    
    #stereo
    for e_range in result['corrupt_range_stereo']:
        for i in range(*e_range):
            labels_stereo[i] = labels_stereo[i] + "*"
    labels_stereo[result['blank_idxs_stereo'][0]] = "[" + labels_stereo[result['blank_idxs_stereo'][0]].strip()
    labels_stereo[result['blank_idxs_stereo'][1]-1] = labels_stereo[result['blank_idxs_stereo'][1]-1].strip() + "]"
    
    
    savepdf = savepdf_pre+"_stereo.pdf"
    with plt.rc_context(rc={"font.family": "serif"}):
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap="RdBu_r",
            vmin=-color_limit,
            vmax=color_limit,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1], 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1], 5)))
        ax.set_yticklabels(labels_stereo)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Impact of restoring state after corrupted input")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        cb.set_label("Normalized indirect effect (NIE)")
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        elif answer is not None:
            # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.

            cb.ax.set_title(f"Corrupt: {' '.join(answer)}", y=-0.16, fontsize=10)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
    




# Utilities for dealing with tokens
def make_inputs(mt, prompts, labels, subject=None, device="cuda", blank_idxs=None):
    if any(k in mt.model_name.lower() for k in ("gpt", "olmo", "pythia")):
        # these tokenizers add no BOS; glue the start marker (StereoSet scoring convention)
        bos = mt.tokenizer.bos_token if mt.tokenizer.bos_token is not None else mt.tokenizer.eos_token
        prompts = [bos + p for p in prompts]
        labels = [bos + p for p in labels]
    
    inputs = mt.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
            # truncation=True
    )
    inputslabels = mt.tokenizer(
            labels,
            padding=True,
            return_tensors="pt",
            # truncation=True
    )
    if inputs['input_ids'].size()[1] != inputslabels['input_ids'].size()[1]:
        print("Prompt and label token lengths differ")
        return None, None, None, None

    # assert inputs['input_ids'].size()[1] == inputslabels['input_ids'].size()[1], "inputs and labels should have the same length"

    subject_range = []
    for subj in subject or []:
        sr = find_token_range(mt.tokenizer, inputs["input_ids"][0], subj, prompts[0])
        if sr is None:
            print(f"Subject {subj!r} was not found as a complete word in {prompts[0]!r}")
            return None, None, None, None
        subject_range.append(sr)

    if blank_idxs is None:
        print(f"BLANK span was not verified for {prompts[0]!r}")
        return None, None, None, None   # no verified span -> skip sample
    blank_start, blank_end = blank_idxs
    if blank_start == 0:
        print(f"BLANK starts at token 0 and cannot be scored by a causal LM: {prompts[0]!r}")
        return None, None, None, None
    if not 0 < blank_start < blank_end <= inputs['input_ids'].shape[1]:
        print(f"BLANK span {blank_idxs} is outside the tokenized input: {prompts[0]!r}")
        return None, None, None, None

    inputs['labels'] = torch.full_like(inputs['input_ids'], -100)
    inputs['labels'][:, blank_start:blank_end] = inputs['input_ids'][:, blank_start:blank_end]
    blank_token_idxs = blank_idxs       # from StereoSetDataset.word_span
    return inputs.to(device), subject_range, blank_token_idxs, inputslabels




def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def find_complete_word(text, substring):
    if substring is None:
        return None
    return re.search(rf"(?<!\w){re.escape(substring)}(?!\w)", text)


def find_token_range(tokenizer, token_array, substring, text=None):
    """Locate the first complete-word occurrence and return its token range [start, end)."""
    if substring is None:
        return None

    if text is not None:
        match = find_complete_word(text, substring)
        if match is None:
            return None
        try:
            encoded = tokenizer(text, return_offsets_mapping=True)
            encoded_ids = encoded["input_ids"]
            token_ids = token_array.tolist() if hasattr(token_array, "tolist") else list(token_array)
            if encoded_ids != token_ids:
                return None
            token_idxs = [
                idx for idx, (start, end) in enumerate(encoded["offset_mapping"])
                if start < match.end() and end > match.start()
            ]
            if not token_idxs:
                return None
            return (token_idxs[0], token_idxs[-1] + 1)
        except (NotImplementedError, TypeError):
            pass

    # Slow-tokenizer fallback: retain the existing decode mapping, but match a
    # complete word instead of an arbitrary substring.
    toks = decode_tokens(tokenizer, token_array)
    whole_string = "".join(toks)
    match = find_complete_word(whole_string, substring)
    if match is None:
        return None

    loc = 0
    tok_start, tok_end = None, None
    for i, t in enumerate(toks):
        loc += len(t)
        if tok_start is None and loc > match.start():
            tok_start = i
        if tok_end is None and loc >= match.end():
            tok_end = i + 1
            break
    if tok_start is None or tok_end is None:
        return None
    return (tok_start, tok_end)




def collect_embedding_std(mt, subjects, device="cuda"):
    alldata = []
    with torch.no_grad():
        for s in tqdm(subjects, desc="Collect Embeddings"):
            inp = mt.tokenizer(
                    [s],
                    padding=True,
                    return_tensors="pt"
            )

            inp.to(device)
            with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
                mt.model(**inp)
                alldata.append(t.output[0])     # t.output (batch_size, seq_len, emb_size)
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()  # the standard deviation over embeddings of all subjects
    return noise_level






if __name__ == "__main__":
    main()
