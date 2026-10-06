# AdaCrowd: Code and Proofs

Supplementary material for our paper.

- `code/`: AdaCrowd implementation and a LabelMe example.
- `proofs/`: Lean formalization of the manuscript's Optimization Interpretation section.

## Run the example

Requires Python 3.10+. From the package root:

```bash
cd code
python -m pip install -r requirements.txt
python train.py
```

The example runs 10 annotation-selection rounds with 2 training epochs per round on CPU. It reports selected annotations, training loss, and test accuracy. Add `--device cuda:0` to use CUDA.

The included LabelMe sample contains 128 training instances, 32 test instances, 8 classes, and 166 annotations from 8 workers. VGG16 features are included. This sample demonstrates the method's workflow; it does not reproduce the paper's full experimental results.

## Build the proofs

Requires the [Elan toolchain manager](https://github.com/leanprover/elan). Lean 4.19.0 and mathlib v4.19.0 are specified by the project files. From the package root:

```bash
cd proofs
lake exe cache get
lake build
```

The first build requires internet access to obtain the toolchain and dependencies.

Main results are in `proofs/AdaCrowd/Stochastic.lean`, under `AdaCrowd.General.StochasticRun`:

| Declaration | Result |
| --- | --- |
| `stationarity` | The manuscript's non-convex stationarity bound |
| `stationarity_sharp` | Stronger bound with bias coefficient 1 instead of 2 |
| `uniform_bias_floor` | Explicit finite-horizon bound with a residual bias term |
| `budget_bias` | Budget–bias corollary for deterministic subset sizes |
| `budget_bias_random` | Random-budget extension retaining expectation around reciprocal subset size |
