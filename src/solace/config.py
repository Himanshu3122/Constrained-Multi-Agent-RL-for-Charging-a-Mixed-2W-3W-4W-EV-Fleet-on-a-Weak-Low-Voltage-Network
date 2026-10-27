"""Experiment configuration (YAML + Pydantic).

Values marked ``# paper`` come from the manuscript (Tables II and III,
Sections III-VI). Values marked ``# assumption`` are engineering choices
documented in docs/assumptions.md.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, model_validator

VehicleClass = Literal["2W", "3W", "4W"]
VEHICLE_CLASSES: tuple[str, ...] = ("2W", "3W", "4W")


class VehicleClassSpec(BaseModel):
    """Per-class fleet parameters (paper Table II)."""

    count: int = Field(ge=0)
    battery_kwh: tuple[float, float]
    chemistry: Literal["LFP", "NMC"]
    charger_kw: list[float]
    mean_dwell_h: float = Field(gt=0)
    v2g_capable: bool
    mean_energy_kwh: float = Field(gt=0)
    sessions_per_day: float = Field(default=1.0, gt=0)  # assumption (A3)
    arrival_modes: list[tuple[float, float, float]]  # (weight, mean hour, std hours), assumption (A4)


def _default_fleet() -> dict[str, VehicleClassSpec]:
    return {
        "2W": VehicleClassSpec(
            count=620, battery_kwh=(2.0, 4.0), chemistry="LFP", charger_kw=[1.0, 3.3],
            mean_dwell_h=2.6, v2g_capable=False, mean_energy_kwh=2.0,
            arrival_modes=[(0.7, 18.0, 1.5), (0.3, 13.0, 2.0)],
        ),
        "3W": VehicleClassSpec(
            count=220, battery_kwh=(5.0, 10.0), chemistry="LFP", charger_kw=[3.3, 7.4],
            mean_dwell_h=3.4, v2g_capable=True, mean_energy_kwh=6.0,
            arrival_modes=[(0.5, 14.0, 3.0), (0.5, 21.0, 1.5)],
        ),
        "4W": VehicleClassSpec(
            count=160, battery_kwh=(30.0, 60.0), chemistry="NMC", charger_kw=[7.4, 30.0],
            mean_dwell_h=8.1, v2g_capable=True, mean_energy_kwh=15.0,
            arrival_modes=[(0.8, 20.0, 1.5), (0.2, 10.0, 1.5)],
        ),
    }


class FeederConfig(BaseModel):
    """Distribution network. Synthetic topology parameters are assumptions (A1)."""

    source: Literal["synthetic", "json"] = "synthetic"
    json_path: str | None = None
    n_transformers: int = Field(default=4, ge=1)  # paper
    transformer_kva: float = Field(default=630.0, gt=0)  # paper
    lv_buses_per_transformer: int = Field(default=62, ge=2)  # paper: 248 LV buses in total
    households: int = Field(default=1340, ge=1)  # paper
    v_base_kv: float = 0.415  # paper (LV line-to-line)
    laterals_per_transformer: int = Field(default=4, ge=1)  # assumption
    segment_r_ohm: float = 0.012  # assumption: ~40 m of LV cable per segment
    segment_x_ohm: float = 0.0035  # assumption
    household_pf: float = Field(default=0.95, gt=0, le=1)  # assumption
    v_min: float = 0.95  # paper
    v_max: float = 1.05  # paper
    polygon_sides: int = Field(default=16, ge=4)  # paper


class TariffConfig(BaseModel):
    """Three-band time-of-day tariff. Only the 1.2x peak ratio is from the paper."""

    normal_inr_per_kwh: float = 7.0  # assumption
    peak_multiplier: float = 1.2  # paper
    offpeak_multiplier: float = 0.8  # assumption
    peak_hours: tuple[int, int] = (18, 22)  # assumption, [start, end)
    offpeak_hours: tuple[int, int] = (22, 6)  # assumption, wraps midnight
    v2g_export_multiplier: float = 1.0  # assumption: discharge credited at the import tariff


class DataConfig(BaseModel):
    n_days: int = Field(default=365, ge=1)  # paper
    split: tuple[int, int, int] = (300, 35, 30)  # paper: train / validation / test days
    dt_minutes: int = 15  # paper
    steps_per_day: int = 96  # paper
    episode_start_hour: int = 8  # assumption (A5): 08:00 -> 08:00 next day
    household_peak_kw: float = 1.4  # assumption
    base_load_noise: float = 0.08  # assumption
    pv_kw_per_transformer: float = 60.0  # assumption
    acn_path: str | None = None  # optional ACN-Data export for 4W sessions
    irradiance_path: str | None = None  # optional hourly irradiance CSV
    tariff: TariffConfig = TariffConfig()

    @model_validator(mode="after")
    def _check(self) -> DataConfig:
        if sum(self.split) > self.n_days:
            raise ValueError(f"split {self.split} needs {sum(self.split)} days but n_days={self.n_days}")
        if self.steps_per_day * self.dt_minutes != 24 * 60:
            raise ValueError("steps_per_day * dt_minutes must equal 24 h")
        return self


class IntentSamplingConfig(BaseModel):
    """How synthetic session constraints are drawn for training (assumption A7)."""

    p_v2g_opt_in: float = 0.4
    soc_floor_range: tuple[float, float] = (0.3, 0.6)
    max_discharge_frac_range: tuple[float, float] = (0.05, 0.2)
    p_early_deadline: float = 0.2
    p_priority_cost: float = 0.25
    p_priority_urgent: float = 0.1


class DegradationChemistry(BaseModel):
    """Semi-empirical cycle + calendar model, paper eq. (7). Parameters are assumptions (A8)."""

    B: float
    Ea_j_per_mol: float
    alpha_j_per_mol_per_c: float
    z: float
    calendar_per_day: float  # fractional capacity loss per day at 50% SoC
    calendar_soc_slope: float  # multiplicative increase per unit SoC above 0.5


class DegradationConfig(BaseModel):
    temperature_c: float = 30.0  # assumption
    battery_cost_inr_per_kwh: float = 12000.0  # assumption
    end_of_life_loss: float = 0.2  # assumption: 20% capacity fade = end of life
    initial_equivalent_cycles: float = 500.0  # assumption A8: batteries are not new at the start of a day
    chemistries: dict[str, DegradationChemistry] = {
        # Functional form of Wang et al. (2011); coefficients must be checked against the source.
        "LFP": DegradationChemistry(B=31630.0, Ea_j_per_mol=31700.0, alpha_j_per_mol_per_c=-370.3, z=0.55,
                                    calendar_per_day=2.0e-5, calendar_soc_slope=1.0),
        "NMC": DegradationChemistry(B=31630.0, Ea_j_per_mol=31700.0, alpha_j_per_mol_per_c=-370.3, z=0.55,
                                    calendar_per_day=4.0e-5, calendar_soc_slope=1.5),
    }


class EnvConfig(BaseModel):
    use_projection: bool = True  # paper: Layer 3
    eta_charge: float = 0.95  # assumption
    eta_discharge: float = 0.95  # assumption
    beta_projection: float = 0.05  # paper (Table III), applied to ||a_hat - a_tilde||^2
    unmet_penalty_inr_per_kwh: float = 30.0  # assumption (A9): soft deadline term
    violation_penalty_inr: float = 0.0  # used only by penalty-based baselines
    reward_scale: float = 1e-3  # assumption: INR -> reward units
    ac_violation_tol: float = 1e-4


class AgentConfig(BaseModel):
    actor_hidden: list[int] = [128, 128]  # paper: 2x128, tanh
    critic_type: Literal["gat_mf", "concat", "local"] = "gat_mf"
    gat_heads: int = 4  # paper
    gat_hidden: int = 128  # paper
    mean_field_dim: int = 64  # paper
    use_mean_field: bool = True
    init_log_std: float = -0.5  # assumption


class TrainingConfig(BaseModel):
    lr_actor: float = 3e-4  # paper
    lr_critic: float = 5e-4  # paper
    gamma: float = 0.99  # paper
    gae_lambda: float = 0.95  # paper
    clip_eps: float = 0.2  # paper
    entropy_coef: float = 0.01  # paper
    ppo_epochs: int = 10  # paper
    minibatches: int = 4  # paper
    n_envs: int = 32  # paper
    rounds: int = 300  # paper (R)
    local_iters: int = 5  # paper (E)
    aggregators_per_transformer: int = 2  # paper (8 aggregators for 4 transformers)
    federated: bool = True
    dual_lr: float = 1e-3  # original paper value; inert with one update per round (audit C3)
    dual_normalize: bool = False  # revision: divide the violation by the budget before the dual step
    lambda_max: float = 100.0  # revision: cap on the multipliers
    d_fairness: float = 0.06  # paper: 1 - Jain <= 0.06
    d_degradation_rel: float = 1.1  # paper: 1.1x no-V2G degradation
    use_fairness_constraint: bool = True
    use_degradation_constraint: bool = True
    projection_gradient: bool = True  # paper: backpropagate through eq. (11)
    projection_grad_batch: int | None = None  # revision: subsample per minibatch for the projection term (None = all)
    max_grad_norm: float = 0.5  # assumption
    device: str = "cpu"


class IntentConfig(BaseModel):
    parser: Literal["rule", "hf"] = "rule"
    hf_model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # paper (weights not distributed)
    hf_adapter_path: str | None = None
    max_retries: int = 2  # paper
    headroom_margin: float = 0.05  # assumption
    sampling: IntentSamplingConfig = IntentSamplingConfig()


class EvaluationConfig(BaseModel):
    seeds: list[int] = [0, 1, 2, 3, 4]  # paper: five seeds
    methods: list[str] = ["uncontrolled", "tou", "bandits", "solace"]
    max_test_days: int | None = None


class ExperimentConfig(BaseModel):
    name: str = "full"
    scenario_set: str | None = None  # preprocessed scenario directory name; defaults to ``name``
    seed: int = 0
    data_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"
    feeder: FeederConfig = FeederConfig()
    fleet: dict[str, VehicleClassSpec] = Field(default_factory=_default_fleet)
    data: DataConfig = DataConfig()
    degradation: DegradationConfig = DegradationConfig()
    env: EnvConfig = EnvConfig()
    agent: AgentConfig = AgentConfig()
    training: TrainingConfig = TrainingConfig()
    intent: IntentConfig = IntentConfig()
    evaluation: EvaluationConfig = EvaluationConfig()

    @property
    def scenario_dir(self) -> Path:
        return Path(self.data_dir) / (self.scenario_set or self.name)

    @property
    def n_aggregators(self) -> int:
        return self.feeder.n_transformers * self.training.aggregators_per_transformer

    @property
    def dt_hours(self) -> float:
        return self.data.dt_minutes / 60.0


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    parent = raw.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        raw = _deep_merge(_load_yaml(parent_path), raw)
    return raw


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load a YAML config (supporting ``extends:``) and validate it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config file not found: {path}")
    raw = _load_yaml(path)
    if overrides:
        raw = _deep_merge(raw, overrides)
    return ExperimentConfig.model_validate(raw)


def dump_config(cfg: ExperimentConfig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.model_dump(mode="json"), fh, sort_keys=False)
