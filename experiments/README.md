# Appendix experiments

These scripts produce the additional experiments of the paper's appendix. Run them from the repository root, with the three CSV files in `data/` (see [`data/README.md`](../data/README.md)); each writes its raw numbers to `results/` and, where the paper has a figure, a PDF and PNG to `figures/`.

| Script | Paper | Output |
|---|---|---|
| `experiment_scale.py` | Appendix E.1, Table S1 (Stage 1 on the full datasets) | `results/scale_<dataset>.json` |
| `experiment_spectrum.py` | Appendix E.3, Table S3 (Theorems 1 and 2 checked on the full spectra) | `results/spectrum.json` |
| `experiment_diffusion_time.py` | Appendix E.3, Fig. S1 (behavior across diffusion times) | `results/sweep_time_<dataset>.json`, `figures/fig_time` |
| `experiment_delta.py` | Appendix E.4.1, Fig. S2 (fairness tolerance $\delta$) | `results/sweep_delta.json`, `figures/fig_delta` |
| `experiment_k_knn.py` | Appendix E.4.2–E.4.3, Figs. S3 and S4 ($k$ and $k_{\mathrm{nn}}$) | `results/sweep_k.json`, `results/sweep_knn.json`, `figures/fig_k`, `figures/fig_knn` |
| `experiment_gamma.py` | Appendix E.4.4, Fig. S5 (vertex-measure parameter $\gamma$, including the boundary $\gamma = 1$ and the ratio $\max\pi / \min\pi$) | `results/sweep_gamma.json`, `figures/fig_gamma` |
| `experiment_groups.py` | Appendix E.4.5, Fig. S6 and Table S4 (number of protected groups, Census) | `results/groups_census.json`, `figures/fig_groups` |

The main comparison (Tables 2 and 3) is `main_pipeline.py` and the synthetic directed networks (Appendix E.2, Table S2) are `experiment_synthetic.py`, both in the repository root. The figures as they appear in the paper are in [`assets/figures/`](../assets/figures).

```bash
python experiments/experiment_delta.py                        # all three datasets
DATASET=Diabetes python experiments/experiment_delta.py       # a subset (comma-separated)
python experiments/experiment_scale.py Census Bank            # Table S1, chosen datasets
DATA_DIR=/path/to/csvs python experiments/experiment_gamma.py # CSV files elsewhere
SWEEPS=knn python experiments/experiment_k_knn.py             # only the k_nn sweep
```

All settings follow the paper: one run per setting with seed 42, $n = 3{,}000$ stratified subsamples (except `experiment_scale.py`), $k = 7$, $\delta = 0.05$, $\gamma = 0.5$, elbow-selected $k_{\mathrm{nn}}$ (12 / 10 / 12 for Diabetes / Census / Bank), $t_{\max} = 40$ and $d = \lceil\sqrt n\rceil$.

**Self-contained by design.** Apart from `experiment_spectrum.py`, which imports `src/`, every script carries its own copy of the pipeline (data cleaning, P-RW operator, entropy-based diffusion time, the $\pi$-orthogonal projector $\Pi_\pi$, and the LP → count-rounding → transportation solver), so it can be run on its own or pasted into a single notebook cell. The computation is the one in `src/`; the only implementation difference is that these scripts solve the per-group transportation problems with HiGHS as linear programs, while `src/fair_assignment.py` uses an equivalent linear assignment. Both return an optimal transportation solution.

**Run time.** The sweeps solve one fair-assignment problem with $nk$ variables per setting and alternation. For one dataset (Diabetes) on a 2-core machine: about 2 minutes for `experiment_spectrum.py`, 6 minutes for `experiment_scale.py` ($n = 70{,}000$), 10–20 minutes for the $\delta$ and diffusion-time sweeps, and 20–40 minutes for the $k$, $k_{\mathrm{nn}}$ and $\gamma$ sweeps. `experiment_spectrum.py` forms dense $n \times n$ matrices and is meant for $n = 3{,}000$.
