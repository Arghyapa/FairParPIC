# Fair Parametrized Power-Iteration Clustering (Fair ParPIC)

This repository contains the official Python implementation of **Fair ParPIC: Fair Power-Iteration Clustering on Directed Graphs** (under review at AISTATS 2027).

Fair ParPIC is a two-stage fair clustering method for **directed graphs** with provable guarantees. It builds on Parametrized Power-Iteration Clustering (ParPIC), which clusters a digraph by power iteration of a reversible random walk parametrized by a vertex measure, without symmetrizing the graph and without any eigendecomposition. Fair ParPIC makes this pipeline fair with respect to a sensitive attribute (gender, marital status, race, ...) and **certifies** the balance of every cluster it returns.

## 📖 Table of Contents
* [Overview](#-overview)
* [Key Contributions](#-key-contributions)
* [Repository Structure](#-repository-structure)
* [Installation](#️-installation)
* [Usage](#-usage)
* [Experimental Results](#-experimental-results)
* [Sensitivity Analysis](#-sensitivity-analysis)
* [Citation](#-citation)

## 🔬 Overview

Unconstrained clustering has no incentive to avoid clusters composed almost entirely of one group whenever group membership is correlated with the similarity structure. Fair spectral methods address this on **undirected** graphs, but many similarity graphs are naturally **directed** (in a $k_{\mathrm{nn}}$-NN graph, "$j$ is among $i$'s nearest neighbours" is not symmetric; citations and hyperlinks are directed by nature).

Fair ParPIC works as follows:

1. Build the directed $k_{\mathrm{nn}}$-NN digraph (or take a given digraph $W$) and its natural random walk $P = D_{\mathrm{out}}^{-1}W$.

2. Turn it into ParPIC's reversible **P-RW operator** $P_{(\nu)} = (D_\nu + D_\xi)^{-1}(D_\nu P + P^\top D_\nu)$, with the degree-based vertex measure $\nu_\gamma$ and $\xi = P^\top\nu$.

3. Select the diffusion time $t^\star$ at the knee of the estimated row-entropy curve $\widehat H(t)$.

4. **Stage 1 — Fair projection.** Run the power iteration and project every step onto the null space of the centered group-indicator matrix $F$, $F(i,g) = \mathbb{I}\{i \in S_g\} - \alpha_g$:

$$ Z^{(\tau)} = \Pi_F\, P_{(\nu)}\, Z^{(\tau-1)}, \qquad \Pi_F = I_n - F(F^\top F)^{-1}F^\top, \qquad \tau = 1,\dots,t^\star .$$

   This enforces $F^\top Z = 0$ exactly, i.e. **centroid parity**: all groups have the same mean in the embedding.

5. **Stage 2 — Fair assignment.** Centroid parity does *not* imply balanced clusters (fairness can be lost in the final $k$-means step), so $k$-means is replaced by an alternating fair assignment that, for fixed centres $\mu_c$, solves

$$ (\mathrm{P}_\delta):\ \min_{x \in \{0,1\}^{n\times k}} \sum_{i,c} x_{ic}\|z_i-\mu_c\|^2 \ \ \text{s.t.}\ \ \sum_c x_{ic}=1,\ \ (1-\delta)\alpha_g \le \frac{|C_c\cap S_g|}{|C_c|} \le \frac{\alpha_g}{1-\delta},\ \ |C_c|\ge 1,$$

   with a scalable LP → count-rounding → transportation solver that satisfies the share bounds **exactly** and reports a certified optimality gap.

**The Fair ParPIC pipeline:**
Below is the full pipeline on a small synthetic example ($n = 180$, $k = 3$, $G = 2$, $\delta = 0.05$): the projection makes the group centroids coincide, $k$-means on the projected embedding still yields unfair group shares, and the fair assignment brings every cluster into the allowed range.

> ![Fair ParPIC pipeline](assets/images/pipeline.png)

**The balance certificate (Theorem 1):** every clustering returned by Fair ParPIC satisfies

$$ (1-\delta)\,\alpha_{\min} \;\le\; \mathrm{balance}(C) = \min_{c}\min_{g} \frac{|C_c\cap S_g|}{|C_c|} \;\le\; \alpha_{\min}, $$

*(where $\alpha_{\min}$ is the smallest group proportion and $\delta$ the user-set tolerance; the upper bound holds for every clustering, so the guarantee is within a factor $1-\delta$ of the best achievable balance)*.

## ✨ Key Contributions

* **Fairness inside an eigendecomposition-free iteration:** the projection is applied at every power-iteration step; the iterates form a power iteration of $T = \Pi_F P_{(\nu)}|_{N_F}$ on an invariant subspace and satisfy $F^\top Z = 0$ to machine precision.

* **The discretization gap:** we prove that centroid parity of the embedding does not transfer to the $k$-means labels — the same gap affects fair spectral methods such as FairSC and FairDen.

* **Certified fair labels:** every feasible assignment satisfies $(1-\delta)\alpha_{\min} \le \mathrm{balance} \le \alpha_{\min}$; we also prove feasibility conditions, minimal intervention (the assignment departs from $k$-means only where a share bound would be violated), and monotone convergence of the alternation.

* **Fairness made cheaper:** the projection lowers the *price of fairness* (relative increase in within-cluster sum of squares) of the exact assignment, e.g. from 2.1% to 0.7% on Bank Marketing.

## 📂 Repository Structure

The codebase is organized as follows:

```text
├── LICENSE
├── README.md
├── requirements.txt
├── main_pipeline.py          # Real-data experiment (Tables 2 and 3 of the paper)
├── experiment_synthetic.py   # Synthetic directed networks (Appendix E.2, Table S2)
├── run_all.sh                # Runs the full experimental pipeline
├── src/
│   ├── fair_parpic.py        # FairParPIC estimator; ParPIC / Fair-Projected ParPIC embeddings
│   ├── parpic.py             # k-NN digraph, P-RW operator, entropy-based diffusion time
│   ├── fairness.py           # Group-indicator matrix F, projector Pi_F, balance objective
│   ├── fair_assignment.py    # Fair assignment (P_delta): decomposition & MILP solvers, fair k-means
│   ├── model_selection.py    # Elbow selection of k_nn and k
│   ├── metrics.py            # Price of fairness, silhouette, substantive clusters
│   └── data.py               # Loading / cleaning of Diabetes, UCI Census and Bank Marketing
├── notebooks/
│   └── fair_parpic.ipynb     # Original development notebook (exact MILP solver, n = 10,000)
├── data/
│   └── README.md             # Where to download the three datasets
└── assets/images/            # Figures used in this README
```

## ⚙️ Installation

To run the pipeline, ensure you have Python ≥ 3.9 installed along with the required dependencies. We recommend setting up a virtual environment.

```bash
git clone https://github.com/<your-username>/FairParPIC.git
cd FairParPIC
pip install -r requirements.txt
```

The only dependencies are NumPy, SciPy (≥ 1.9, whose `milp` wraps the HiGHS solver), scikit-learn and pandas. Download the three real datasets into `data/` as described in [`data/README.md`](data/README.md).

## 🚀 Usage

**Run the full experimental pipeline:**

```bash
bash run_all.sh
```

**Run the real-data experiment on one dataset:**

```bash
python main_pipeline.py --dataset census --k 7 --n-neighbors 10
python main_pipeline.py --dataset bank --delta 0.1 --solver milp     # other tolerance / exact MILP
```

**Run the synthetic directed-network experiment** (a subset via environment variables):

```bash
python experiment_synthetic.py
TYPES=A LEVELS=none,strong python experiment_synthetic.py
```

**Use Fair ParPIC on your own data:**

```python
from src import FairParPIC

# X: (n, p) feature matrix, s: (n,) sensitive attribute (any labels)
model = FairParPIC(n_clusters=7, n_neighbors=10, delta=0.05)
labels = model.fit_predict(X, s)
print(model.balance_, model.floor_, model.alpha_min_)   # floor <= balance <= alpha_min

# Directed graph given directly (citation, hyperlink, social networks, ...)
labels = FairParPIC(n_clusters=5).fit_predict(adjacency=W, sensitive=s)
```

If no clustering can satisfy the share bounds (e.g. a group has fewer than $k$ members), `InfeasibleFairnessError` is raised instead of silently returning an unfair clustering.

## 📊 Experimental Results

Fair ParPIC was compared with PIC, AdaPIC, the Fair p-Assignment of Bera et al., FairSC, FairDen, unconstrained ParPIC and its Stage 1 alone (FP-ParPIC) on three real datasets ($n = 3{,}000$ stratified subsamples, $k = 7$, $\delta = 0.05$, mean ± standard deviation over five seeds).

### Real-World Datasets

| Method | Diabetes (gender, $G=2$) | Census (gender, $G=2$) | Bank (marital, $G=3$) |
|---|:---:|:---:|:---:|
| PIC | 0.085 ± 0.169 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| AdaPIC | 0.080 ± 0.160 | 0.013 ± 0.025 | 0.000 ± 0.000 |
| Fair p-Assignment | 0.323 ± 0.164 | 0.248 ± 0.017 | 0.088 ± 0.008 |
| FairSC | 0.427 ± 0.013 | 0.130 ± 0.065 | 0.069 ± 0.024 |
| FairDen | 0.420 ± 0.018 | 0.087 ± 0.107 | 0.094 ± 0.010 |
| ParPIC | 0.396 ± 0.052 | 0.130 ± 0.065 | 0.074 ± 0.011 |
| FP-ParPIC (Stage 1) | 0.398 ± 0.054 | 0.130 ± 0.065 | 0.071 ± 0.020 |
| **Fair ParPIC** | **0.446 ± 0.007** | **0.315 ± 0.000** | **0.110 ± 0.001** |
| *Guaranteed floor $(1-\delta)\alpha_{\min}$* | *0.439* | *0.315* | *0.110* |
| *Ceiling $\alpha_{\min}$* | *0.462* | *0.332* | *0.115* |

*Balance (higher is better).* Fair ParPIC attains balance within 5% of the best achievable value in every run, with the same number of substantive clusters as ParPIC.

**Price of fairness and cluster quality.** The projection makes the exact fair assignment cheaper, and the fair assignment barely changes the silhouette score:

| | Diabetes | Census | Bank |
|---|:---:|:---:|:---:|
| PoF — ParPIC + fair assignment | 0.9% ± 1.2% | 2.2% ± 0.8% | 2.1% ± 1.1% |
| PoF — **Fair ParPIC** | **0.8% ± 1.0%** | **1.9% ± 0.7%** | **0.7% ± 0.3%** |
| Silhouette — ParPIC | 0.220 ± 0.018 | 0.110 ± 0.011 | 0.165 ± 0.013 |
| Silhouette — **Fair ParPIC** | 0.218 ± 0.018 | 0.113 ± 0.006 | 0.161 ± 0.011 |

### Stage 1 at Larger Scale

Stage 1 uses only sparse products and a rank-$(G-1)$ correction, so it runs on the full datasets ($k_{\mathrm{nn}} = 10$, $k = 10$, $k$-means discretization, seed 42):

| Dataset | $n$ | ParPIC balance | FP-ParPIC balance | $\max_{i,j}\lvert(F^\top Z)_{ij}\rvert$ | Time (s) |
|---|:---:|:---:|:---:|:---:|:---:|
| Diabetes | 70,000 | 0.417 | **0.427** | 1.8 × 10⁻¹² | 4.4 |
| Census | 48,842 | 0.190 | **0.311** | 5.7 × 10⁻¹³ | 2.3 |
| Bank | 45,211 | 0.068 | **0.073** | 5.6 × 10⁻¹⁴ | 1.6 |

### Synthetic Directed Networks

`experiment_synthetic.py` generates directed stochastic block models (no features, no $k$-NN graph) in which edges point mostly from later to earlier vertices, and a **group signal** $h$ controls how strongly the protected groups shape the links. Balance (guaranteed floor $(1-\delta)\alpha_{\min}$ in brackets) and ARI to the planted clusters, for $h$ = 0 / 0.3 / 0.6 / 0.9:

| Network (floor) | Signal $h$ | ParPIC balance | Fair ParPIC balance | ParPIC ARI | Fair ParPIC ARI |
|---|:---:|:---:|:---:|:---:|:---:|
| Citation, $n$ = 10,000 (0.2850) | 0 / 0.3 / 0.6 / 0.9 | 0.2945 / 0.2957 / 0.0012 / 0.0000 | 0.2943 / 0.2982 / 0.2948 / 0.2852 | 0.895 / 0.916 / 0.396 / 0.280 | 0.897 / 0.918 / 0.912 / 0.829 |
| Popularity, $n$ = 8,000 (0.1425) | 0 / 0.3 / 0.6 / 0.9 | 0.0000 / 0.0000 / 0.0010 / 0.0000 | 0.1429 / 0.1429 / 0.1425 / 0.1427 | 0.491 / 0.453 / 0.270 / 0.137 | 0.549 / 0.383 / 0.544 / 0.354 |
| Hyperlink, $n$ = 6,000 (0.2850) | 0 / 0.3 / 0.6 / 0.9 | 0.1500 / 0.1494 / 0.0000 / 0.0000 | 0.2854 / 0.2852 / 0.2853 / 0.2853 | 1.000 / 1.000 / 0.546 / 0.330 | 0.846 / 0.848 / 0.848 / 0.834 |

Fair ParPIC meets its guarantee on all twelve networks; ParPIC reaches the floor on only two. On the hyperlink networks the planted clustering is itself unfair (balance 0.150), so reaching the floor necessarily moves vertices away from it.

## 🔬 Sensitivity Analysis

Each hyperparameter is varied one at a time ($n = 3{,}000$, seed 42). In every figure, the **top row** shows the balance of Fair ParPIC (blue) and unconstrained ParPIC (orange) against the floor (dotted) and ceiling (solid), and the **bottom row** shows the price of fairness with (green) and without (red) the projection. All 164 fair-assignment runs met the floor.

### 1. Fairness tolerance $\delta$

$\delta$ is the only hyperparameter that enters the guarantee. Fair ParPIC follows the floor $(1-\delta)\alpha_{\min}$ while the constraint is active, and returns its $k$-means clustering at zero price as soon as the floor drops below it (minimal intervention).

> ![Effect of delta](assets/images/sensitivity_delta.png)

### 2. Number of clusters $k$

The balance of unconstrained ParPIC degrades as $k$ grows (reaching 0 on Bank at $k = 15$), whereas Fair ParPIC stays between floor and ceiling for every $k$; the projection slows the growth of the price of fairness.

> ![Effect of k](assets/images/sensitivity_k.png)

### 3. Number of neighbours $k_{\mathrm{nn}}$

The balance of ParPIC changes erratically with the graph, so tuning the graph is not a reliable route to fairness; Fair ParPIC varies by at most 0.002, and the projection lowers the price of fairness in 17 of 18 settings.

> ![Effect of knn](assets/images/sensitivity_knn.png)

### 4. Vertex-measure parameter $\gamma$

Fair ParPIC is on the floor for every $\gamma$; on Bank the projection keeps the price of fairness below 0.74% against up to 2.93% without it.

> ![Effect of gamma](assets/images/sensitivity_gamma.png)

### 5. Number of protected groups $G$ (Census)

With small groups ($G \ge 4$), unconstrained clustering leaves at least one cluster without any member of the smallest group (balance 0), while Fair ParPIC holds its guarantee for every grouping, from $G = 2$ to gender × race ($G = 8$).

> ![Effect of the number of groups](assets/images/sensitivity_groups.png)

<!--## 📜 Citation

If you find this code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{fairparpic2027,
  title={Fair ParPIC: Fair Power-Iteration Clustering on Directed Graphs},
  author={Anonymous},
  booktitle={Submitted to the International Conference on Artificial Intelligence and Statistics (AISTATS)},
  year={2027}
}
```
-->

## 📜 Citation

The paper is currently under double-blind review; citation details will be added after the review process.

This work builds on ParPIC: G. Debaussart-Joniec, H. Sevi, M. Jonckheere and A. Kalogeratos, *Parametrized Power-Iteration Clustering for Directed Graphs*, ICML 2026 (arXiv:2210.00310).
