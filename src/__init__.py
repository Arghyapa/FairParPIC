"""Fair ParPIC: Fair Power-Iteration Clustering on Directed Graphs."""

from .fair_assignment import InfeasibleFairnessError, fair_kmeans
from .fair_parpic import FairParPIC, parpic_embeddings, p_rw_from_graph
from .fairness import (balance_objective, build_fairness_projector,
                       build_group_indicator_matrix)

__all__ = ["FairParPIC", "parpic_embeddings", "p_rw_from_graph", "fair_kmeans",
           "InfeasibleFairnessError", "balance_objective",
           "build_group_indicator_matrix", "build_fairness_projector"]
