"""Layer 3: convex safety projection (paper Section III-D and eq. (11))."""

from solace.safety.projection import (
    ProjectionGroup,
    ProjectionProblem,
    build_transformer_problem,
    differentiable_project,
    project_groups,
    solve_projection,
)

__all__ = [
    "ProjectionGroup",
    "ProjectionProblem",
    "build_transformer_problem",
    "differentiable_project",
    "project_groups",
    "solve_projection",
]
