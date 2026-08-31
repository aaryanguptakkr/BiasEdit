import json
import string

from torch.utils.data import Dataset

from util.globals import *


class StereoSetDataset(Dataset):
    """StereoSet intrasentence samples for causal-LM bias tracing.

    Each item carries the two real sentences (anti/stereo) plus the token span
    of the word that filled BLANK, located by word_span and decode-verified.
    """

    def __init__(self, tokenizer, data_path, model_name):
        super().__init__()
        self.tokenizer = tokenizer
        self.use_empirical_blank = True   # causal-only pipeline

        # gpt2/olmo/pythia inputs get a start marker glued on (see make_inputs),
        # shifting every position right by 1. word_span measures the raw sentence, so:
        #   raw [My][ father][ is][ chief] span (3,4) -> fed [<eot>][My].. span (4,5)
        # llama/gemma: tokenizer adds BOS itself; qwen: none by design -> offset 0.
        self._span_offset = 0
        if any(k in model_name.lower() for k in ("gpt", "olmo", "pythia")):
            marker = tokenizer.bos_token if tokenizer.bos_token is not None else tokenizer.eos_token
            self._span_offset = len(tokenizer(marker, add_special_tokens=False)["input_ids"])

        data = json.load(open(data_path))
        self.data = [{k: d[k] for k in ["id", "target", "bias_type", "context", "data", "subject"]}
                     for d in data]
        print(f"Loaded dataset with {len(self)} elements")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        obj = self.data[item]
        word_idx = next((i for i, w in enumerate(obj["context"].split(" ")) if "BLANK" in w), None)
        if word_idx is None:
            raise Exception("No blank word found.")

        def clean(sentence):
            # Strip punctuation around the fill while preserving internal punctuation.
            return sentence.split(" ")[word_idx].strip(string.punctuation)

        def word_span(sentence):
            """(start, end) token span of the fill word; None (=skip) if unverifiable.

            e.g. "My father is chief operator.", word_idx=3:
                 len(tok("My father is")) = 4, len(tok("My father is chief")) = 5 -> (4, 5)
            """
            words = sentence.split(" ")
            if word_idx >= len(words):
                return None
            prefix = " ".join(words[:word_idx])
            word = clean(sentence)
            # Hyphens/apostrophes are real word-internal characters ("co-worker", "nurse's");
            # any other internal punctuation is a data-entry artifact ("very`quiet", "kind!p").
            if any(c in string.punctuation and c not in "-'" for c in word):
                print(f"Skipping {obj['id']}: fill word {word!r} is a data-entry artifact")
                return None
            sep = " " if word_idx > 0 else ""   # space rides WITH the word (byte-BPE fuses it)
            start = len(self.tokenizer(prefix)["input_ids"])
            end = len(self.tokenizer(prefix + sep + word)["input_ids"])
            full = self.tokenizer(sentence)["input_ids"]

            if end > len(full):
                print(f"Skipping {obj['id']}: cannot verify BLANK span in {sentence!r}")
                return None

            # Some tokenizers emit a standalone whitespace token before the word.
            while start < end and not self.tokenizer.decode(full[start:start + 1]).strip():
                start += 1
            while start < end and not self.tokenizer.decode(full[end - 1:end]).strip():
                end -= 1

            if self.tokenizer.decode(full[start:end]).strip() != word:
                print(f"Skipping {obj['id']}: cannot verify BLANK span in {sentence!r}")
                return None
            # proof ran in raw coords; return in model-input coords (+marker offset)
            return (start + self._span_offset, end + self._span_offset)

        anti = obj["data"]["anti-stereotype"]["sentence"]
        stereo = obj["data"]["stereotype"]["sentence"]
        unrelated = obj["data"]["unrelated"]["sentence"]
        return {
            "id": obj["id"],
            "context": obj["context"],
            "anti": anti,
            "stereo": stereo,
            "unrelated": unrelated,
            "anti_mask": anti,          # interface compat: no masking, = real sentence
            "stereo_mask": stereo,
            "unrelated_mask": unrelated,
            "attribute": {"anti": clean(anti), "stereo": clean(stereo), "unrelated": clean(unrelated)},
            "target": obj["target"],
            "subject": obj["subject"],
            "anti_blank_idxs": word_span(anti),
            "stereo_blank_idxs": word_span(stereo),
        }
