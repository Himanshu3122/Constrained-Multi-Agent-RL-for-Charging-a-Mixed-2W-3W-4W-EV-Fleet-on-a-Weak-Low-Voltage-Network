"""Baselines from paper Section VI-D that are implemented in this repository.

Implemented: uncontrolled, ToU rule, decentralized Thompson-sampling bandits
(approximation), solver-grounded single agent (receding-horizon convex program
with a day-ahead persistence forecast), centralized MPC with perfect foresight,
a fairness-aware variant of that MPC (max-min delivered fraction; added in revision),
and IPPO / MAPPO via the learned-policy trainer (configs/baselines/).
Not implemented: MADDPG, QMIX and the LLM-only scheduler (see
IMPLEMENTATION_STATUS.md).
"""

from solace.baselines.bandits import ThompsonBanditPolicy
from solace.baselines.horizon_qp import CentralizedMPCPolicy, FairMPCPolicy, SolverGroundedPolicy
from solace.baselines.rule_based import Policy, ToUPolicy, UncontrolledPolicy

__all__ = [
    "CentralizedMPCPolicy",
    "FairMPCPolicy",
    "Policy",
    "SolverGroundedPolicy",
    "ThompsonBanditPolicy",
    "ToUPolicy",
    "UncontrolledPolicy",
]
