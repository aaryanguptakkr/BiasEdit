# BiasEdit: Debiasing Stereotyped Language Models via Model Editing


<p align="center">
  <a href="https://arxiv.org/abs/2503.08588">📃 Paper</a> 
  <a href="https://github.com/zjunlp/BiasEdit">💻 Code</a> 
  <a href="https://zjunlp.github.io/project/BiasEdit">🌏 Web</a> 
</p>



<div align=center><img src="fig/BiasEdit_fig1.gif" width="70%"/></div>


**BiasEdit** is an efficient *model editing* method to eliminate stereotyped bias from language models with small editor networks, including a *debiasing loss* to guide edits on partial parameters and a *remaining loss* to maintain the language modeling abilities during editing. Experimental results show BiasEdit' excellent performance on debiasing, modeling ability preservation, and robustness of gender reverse and semantic generality.

## 📌 Table of Contents

- [🛠️ Setup](#1)
- [💻 BiasEdit](#2)
    - [⌚️ Training Editor Networks](#2.1)
    - [🚀 Debiasing with Editor Networks](#2.2)
- [👀 Bias Tracing](#3)
- [📝 Citation](#4)
- [✨ Acknowledgements](#5)

<h2 id="1">🛠️ Setup</h2>

This codebase uses Python 3.9.18. Other versions may work as well.

Create an environment
and install the dependencies:

    $ conda create -n biasedit python=3.9
    $ conda activate biasedit
    (biasedit) $ pip install -r requirements.txt


<h2 id="2">💻 BiasEdit</h2>
<div align=center><img src="fig/BiasEdit_fig2.png" width="80%"/></div>

With [StereoSet](https://aclanthology.org/2021.acl-long.416/), editor networks are trained to generate parameter shifts for debiasing at first. Then, the trained editor networks are used to conduct edits on language models and produce an unbiased model.

<h3 id="2.1">⌚️ Training Editor Networks</h3>

- Formatted datasets with [train](./data/stereoset/train.json)/[dev](./data/stereoset/dev.json)/test (`gender_test.json`, `race_test.json`, `religion_test.json`) splits are in [data/stereoset](./data/stereoset). 
- Configurations are in [config](./config). Partial parameters to be edited are presented in [editor](./config/editor). The configurations, like weights to be edited, are in [model](config/model).
- Experimental scripts are in [scripts](./scripts). All hyper-parameters are in the scripts. Since hyper-parameters have a great effect on hyper-network tuning, higly recommand conducting hyper-paramter tuning.
- For the ablation study on the remaining loss, set `editor.loc_coef=0`.
- Metrics can be found in the training log.


<h3 id="2.2">🚀 Debiasing with Editor Networks</h3>

- Set `eval_only=True`
- Set `data.valid_path` as the path of the test set
- Metrics can be found at the end of the debiasing log, like "Test ------- XXX".
- Experimental scripts are in [scripts](./scripts).


<h2 id="3">👀 Bias Tracing</h2>

Enter [bias_tracing](./bias_tracing).


<h2 id="4">📝 Citation</h2>

If this code or paper is useful, please consider using the following citation:

    @article{xin25BiasEdit,
        title={BiasEdit: Debiasing Stereotyped Language Models via Model Editing},
        author={Xin Xu, Wei Xu, Ningyu Zhang, Julian McAuley},
        year={2025},
        url={https://arxiv.org/pdf/2503.08588}
    }

<h2 id="5">✨ Acknowledgements</h5>

- Thanks for the original code from [MALMEN](https://github.com/ChenmienTan/malmen).
- Thanks for StereoSet and all the baselines from [bias-bench](https://github.com/McGill-NLP/bias-bench).
- For more model editing methods, please try [EasyEdit](https://github.com/zjunlp/EasyEdit).
