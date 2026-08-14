"""Research-only terrain-response and hurdle Item-Elo prototype.

This module deliberately does **not** create another cumulative rating ledger.
It keeps one frozen stable-readiness state and asks how the athlete's response
changes across procedure-conditioned boulder difficulty.  The response model
is an ordered two-hurdle item-response model::

    s_i = s_0 exp(r_i),                     s_i > 0
    a_iz(c) = theta_i + z(procedure)' u_i + x_pre(style)' v_iz
    a_ip(c) = theta_i + z(procedure)' u_i + x_post(style)' v_ip
    P_i(zone j) = logistic((a_iz(c) - d_zj) / s_i)
    P_i(top j | zone j) = logistic((a_ip(c) - d_pj) / s_i)

``theta_i`` is the existing stable readiness frozen before the competition.
``r_i``, ``u_i`` and ``v_i`` are zero-centred, partially pooled athlete
deviations.  Positive ``s_i`` makes success monotonically decrease with hurdle
difficulty, while different ``s_i`` values permit honest crossing response
curves: one athlete may clean easier terrain more reliably while another
retains more probability on exceptionally hard terrain.

Population-wide procedure and style effects are intentionally absent.  When
an item's Elo is inferred from that round's field, population format effects,
route-setting and the common event shock are inseparable from item difficulty.
Only an athlete's *relative* response to confirmed procedure/style is
identified.  A future target must consequently use a pre-event distribution
of procedure-conditioned Item-Elo learned from earlier comparable rounds.

Every post-event zone/post-zone pair receives a joint Item-Elo posterior.  The
posterior used to evaluate athlete ``i`` excludes every observation and frozen
state belonging to ``i``; sparse items remain prior-dominated or censored and
never fall back to the full-field estimate.  Zone-Elo is the event-level 50%
zone threshold.  Post-zone-Elo is the event-level 50% continuation threshold
conditional on zone and is not constrained above Zone-Elo: the finish can be
the easier segment.  Cumulative Top-Elo is derived as the stable readiness
whose two hurdle probabilities multiply to 50%.  Attempts are excluded from
Item-Elo.  In the response likelihood the attempt on which Zone was first
reached is a weak refinement only when at least two athletes share the same
achievement stage on that item.  Aggregate Top tries do not identify how many
conditional post-zone opportunities occurred and are therefore metadata only;
post-zone efficiency requires an explicit opportunity horizon and success
index.  Published rank, score and aggregate totals never form another
likelihood.

For a future event, ``Projection-Elo`` is target-conditioned rather than a new
state.  It is the population-reference Elo that gives the same expected
zone/top achievement after integrating over the frozen pre-event item
difficulty distribution.  The projection carries its target ID, model name
and uncertainty and must never feed back into stable readiness.

The fitting routine is a deterministic empirical-Bayes MAP prototype using
coordinate search and diagonal curvature.  Its leave-one-athlete-out item
posteriors form a weighted composite likelihood, not independent data.  It is
appropriate for synthetic/adversarial checks and a future rolling-origin
challenger audit; it is not production rating code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Mapping, Sequence

import numpy as np


CONFIRMED_PROCEDURES = ("onsight", "flash", "scramble")
ITEM_ELO_SEMANTICS = "procedure_conditioned_event_q50_hurdle_elo_v1"
MODEL_NAME = "terrain_response_item_elo_v1_research"
DEFAULT_RATING_POOL = "global_open_readiness_v5_shadow"
PROJECTION_ESTIMANDS = (
    "endpoint_vector_only",
    "modern_25_10_achievement_without_attempt_penalty",
)
_FORBIDDEN_TARGET_PROVENANCE = (
    "target result",
    "target-result",
    "realised target",
    "realized target",
    "observed target",
)
_EPS = 1.0e-12


class TerrainContractError(ValueError):
    """Raised when chronology or model semantics would be ambiguous."""


def _utc(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise TerrainContractError(f"{label} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TerrainContractError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise TerrainContractError(f"{label} must be finite")
    return number


def _procedure(value: object) -> str:
    text = " ".join(str(value or "").strip().casefold().replace("_", " ").split())
    if "onsight" in text or "on sight" in text:
        return "onsight"
    if "flash" in text:
        return "flash"
    if "scramble" in text:
        return "scramble"
    return "unknown"


def _procedure_codes(value: str) -> np.ndarray:
    # Effect coding: the population reference remains zero and no procedure is
    # assigned an ordinal position.
    return {
        "onsight": np.asarray([1.0, 0.0]),
        "flash": np.asarray([0.0, 1.0]),
        "scramble": np.asarray([-1.0, -1.0]),
    }[value]


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(value, dtype=float)
    output = np.empty_like(array)
    positive = array >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-np.minimum(array[positive], 60.0)))
    exponent = np.exp(np.maximum(array[~positive], -60.0))
    output[~positive] = exponent / (1.0 + exponent)
    return float(output) if output.ndim == 0 else output


def _logsumexp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return float(maximum + math.log(float(np.exp(values - maximum).sum())))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = weights[order]
    cumulative = np.cumsum(ordered_weights)
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(ordered_values[min(max(index, 0), len(ordered_values) - 1)])


def _normal_effective_scale(response_scale: float, normal_sd: float) -> float:
    # Logistic-normal moment approximation for Q-D uncertainty.
    return float(math.sqrt(response_scale**2 + 3.0 * normal_sd**2 / math.pi**2))


def _ordered_stage_probability_grid(
    stable_mean: float,
    stable_sd: float,
    zone_grid: np.ndarray,
    post_zone_grid: np.ndarray,
    response_scale: float,
    stage: str,
    *,
    quadrature_nodes: int = 31,
) -> np.ndarray:
    """Integrate an ordered stage over one shared uncertain stable state.

    The Zone and post-Zone hurdles share the same latent pre-event readiness
    draw. Marginalizing the two hurdles separately and multiplying those
    marginals is incorrect whenever stable readiness is uncertain.
    """

    if stage not in {"no_zone", "zone_only", "top"}:
        raise TerrainContractError("ordered item stage is not declared")
    if quadrature_nodes < 9:
        raise TerrainContractError("stable-state quadrature requires >=9 nodes")
    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_nodes))
    latent = float(stable_mean) + math.sqrt(2.0) * float(stable_sd) * nodes
    weights = weights / math.sqrt(math.pi)
    zone = np.asarray(zone_grid, dtype=float).reshape(-1)
    post = np.asarray(post_zone_grid, dtype=float).reshape(-1)
    p_zone = _sigmoid(
        (latent[:, None] - zone[None, :]) / float(response_scale)
    )
    if stage == "no_zone":
        marginal = weights @ (1.0 - p_zone)
        return np.repeat(marginal[:, None], len(post), axis=1)
    p_post = _sigmoid(
        (latent[:, None] - post[None, :]) / float(response_scale)
    )
    if stage == "top":
        return np.einsum("q,qz,qp->zp", weights, p_zone, p_post)
    return np.einsum("q,qz,qp->zp", weights, p_zone, 1.0 - p_post)


@dataclass(frozen=True)
class ProblemOutcome:
    """One athlete/problem result with the stable state frozen pre-event."""

    athlete_id: str
    competition_id: str
    problem_id: str
    competition_start: datetime
    frozen_at: datetime
    result_available_at: datetime
    stable_mean: float
    stable_sd: float
    procedure: str
    reached_zone: bool
    reached_top: bool
    rating_pool_id: str = DEFAULT_RATING_POOL
    # Ordinal success-attempt fields from a versioned source adapter.  They are
    # not aggregate failed-attempt counts.
    attempts_to_zone: int | None = None
    attempts_to_top: int | None = None
    # Explicit conditional continuation exposure.  Do not derive either field
    # from attempts_to_top: total Top tries do not reveal Zone opportunities.
    post_zone_opportunities: int | None = None
    post_zone_success_index: int | None = None
    failed_attempts: int | None = None
    attempt_cap: int = 8
    pre_zone_style_features: Mapping[str, float] = field(default_factory=dict)
    post_zone_style_features: Mapping[str, float] = field(default_factory=dict)
    style_available_at: datetime | None = None
    published_rank: int | None = None
    published_score: float | None = None

    def __post_init__(self) -> None:
        if not self.athlete_id or not self.competition_id or not self.problem_id:
            raise TerrainContractError("athlete_id, competition_id and problem_id are required")
        if not str(self.rating_pool_id).strip():
            raise TerrainContractError("rating_pool_id is required")
        start = _utc(self.competition_start, "competition_start")
        frozen = _utc(self.frozen_at, "frozen_at")
        available = _utc(self.result_available_at, "result_available_at")
        if frozen >= start:
            raise TerrainContractError("stable readiness must be frozen before competition_start")
        if available < start:
            raise TerrainContractError("result cannot be available before competition_start")
        if _procedure(self.procedure) not in CONFIRMED_PROCEDURES:
            raise TerrainContractError("procedure must be confirmed onsight, flash or scramble")
        _finite(self.stable_mean, "stable_mean")
        if _finite(self.stable_sd, "stable_sd") <= 0.0:
            raise TerrainContractError("stable_sd must be positive")
        if self.reached_top and not self.reached_zone:
            raise TerrainContractError("top must imply zone")
        for label, value, achieved in (
            ("attempts_to_zone", self.attempts_to_zone, self.reached_zone),
            ("attempts_to_top", self.attempts_to_top, self.reached_top),
        ):
            if value is not None and (not achieved or int(value) < 1):
                raise TerrainContractError(f"{label} is valid only for an achieved hurdle")
        if self.failed_attempts is not None and self.failed_attempts < 0:
            raise TerrainContractError("failed_attempts cannot be negative")
        if self.attempt_cap < 1:
            raise TerrainContractError("attempt_cap must be a positive predeclared horizon")
        for label, value in (
            ("attempts_to_zone", self.attempts_to_zone),
            ("attempts_to_top", self.attempts_to_top),
        ):
            if value is not None and value > self.attempt_cap:
                raise TerrainContractError(f"{label} exceeds the predeclared attempt_cap")
        if self.post_zone_opportunities is not None:
            if not self.reached_zone:
                raise TerrainContractError(
                    "post_zone_opportunities require an achieved zone"
                )
            if not 1 <= int(self.post_zone_opportunities) <= self.attempt_cap:
                raise TerrainContractError(
                    "post_zone_opportunities must be a positive predeclared horizon "
                    "within attempt_cap"
                )
        if self.post_zone_success_index is not None:
            if not self.reached_top or self.post_zone_opportunities is None:
                raise TerrainContractError(
                    "post_zone_success_index requires a top and an explicit "
                    "post-zone opportunity horizon"
                )
            if not 1 <= int(self.post_zone_success_index) <= int(
                self.post_zone_opportunities
            ):
                raise TerrainContractError(
                    "post_zone_success_index exceeds the post-zone opportunity horizon"
                )
        for segment, features in (
            ("pre-zone", self.pre_zone_style_features),
            ("post-zone", self.post_zone_style_features),
        ):
            for name, value in features.items():
                if not str(name).strip():
                    raise TerrainContractError("style feature names cannot be empty")
                _finite(value, f"{segment} style feature {name}")
        if self.pre_zone_style_features or self.post_zone_style_features:
            if self.style_available_at is None:
                raise TerrainContractError("style_available_at is required with style features")
            if _utc(self.style_available_at, "style_available_at") > available:
                raise TerrainContractError("style tags were not available with the result evidence")

    @property
    def confirmed_procedure(self) -> str:
        return _procedure(self.procedure)

    @property
    def stage(self) -> str:
        return "top" if self.reached_top else "zone" if self.reached_zone else "no_zone"


@dataclass(frozen=True)
class FrozenItemPrior:
    """Pre-event anchor for one zone/conditional-post-zone posterior."""

    competition_id: str
    problem_id: str
    competition_start: datetime
    as_of: datetime
    zone_mean: float
    zone_sd: float
    post_zone_mean: float
    post_zone_sd: float
    correlation: float = 0.35
    provenance: str = "historical comparable procedure-conditioned items"
    semantics: str = ITEM_ELO_SEMANTICS
    rating_pool_id: str = DEFAULT_RATING_POOL
    reference_response_scale: float = 155.0

    def __post_init__(self) -> None:
        if _utc(self.as_of, "item prior as_of") >= _utc(
            self.competition_start, "competition_start"
        ):
            raise TerrainContractError("item prior must be frozen before competition_start")
        if self.semantics != ITEM_ELO_SEMANTICS:
            raise TerrainContractError("item prior has incompatible Elo semantics")
        if not str(self.rating_pool_id).strip():
            raise TerrainContractError("item prior rating_pool_id is required")
        if _finite(self.reference_response_scale, "reference_response_scale") <= 0.0:
            raise TerrainContractError("reference_response_scale must be positive")
        for label in ("zone_mean", "post_zone_mean"):
            _finite(getattr(self, label), label)
        for label in ("zone_sd", "post_zone_sd"):
            if _finite(getattr(self, label), label) <= 0.0:
                raise TerrainContractError(f"{label} must be positive")
        if not -0.95 <= self.correlation <= 0.95:
            raise TerrainContractError("item-prior correlation must be in [-.95, .95]")
        if not str(self.provenance).strip():
            raise TerrainContractError("item prior provenance is required")


@dataclass(frozen=True)
class HurdleItemElo:
    mean: float
    sd: float
    lower: float
    upper: float
    status: str
    successes: int
    opportunities: int


@dataclass(frozen=True)
class LeaveOneOutItemElo:
    competition_id: str
    problem_id: str
    procedure: str
    excluded_athlete_id: str
    zone: HurdleItemElo
    post_zone: HurdleItemElo
    top: HurdleItemElo
    zone_post_covariance: float
    peer_count: int
    calibrated_at: datetime
    semantics: str = ITEM_ELO_SEMANTICS
    rating_pool_id: str = DEFAULT_RATING_POOL
    reference_response_scale: float = 155.0
    source: str = "leave_one_athlete_out_zone_post_posterior"

    def __post_init__(self) -> None:
        if self.semantics != ITEM_ELO_SEMANTICS:
            raise TerrainContractError("Item-Elo has incompatible semantics")
        if not str(self.rating_pool_id).strip():
            raise TerrainContractError("Item-Elo rating_pool_id is required")
        if _finite(self.reference_response_scale, "reference_response_scale") <= 0.0:
            raise TerrainContractError("reference_response_scale must be positive")
        if self.peer_count < 0:
            raise TerrainContractError("peer_count cannot be negative")


@dataclass(frozen=True)
class TerrainResponseConfig:
    development_end: datetime
    response_scale: float = 155.0
    response_scale_min: float = 65.0
    response_scale_max: float = 360.0
    log_scale_prior_sd: float = 0.40
    athlete_procedure_prior_sd: float = 105.0
    athlete_style_prior_sd: float = 90.0
    item_prior_reference_center: float = 2000.0
    item_prior_sd: float = 360.0
    zone_prior_offset: float = -90.0
    top_prior_offset: float = 120.0
    item_grid_size: int = 91
    credible_mass: float = 0.90
    attempt_information_weight: float = 0.20
    composite_weight_exponent: float = 0.50
    minimum_athlete_items: int = 6
    minimum_difficulty_spread: float = 120.0
    minimum_style_spread: float = 0.35
    style_names: tuple[str, ...] = ()
    style_center: float = 1.5
    style_scale: float = 1.5
    coordinate_iterations: int = 5
    coordinate_points: int = 17
    projection_draws: int = 1200
    projection_seed: int = 1701

    def __post_init__(self) -> None:
        _utc(self.development_end, "development_end")
        for label in (
            "response_scale",
            "response_scale_min",
            "response_scale_max",
            "log_scale_prior_sd",
            "athlete_procedure_prior_sd",
            "athlete_style_prior_sd",
            "item_prior_sd",
            "style_scale",
            "minimum_difficulty_spread",
            "minimum_style_spread",
        ):
            if _finite(getattr(self, label), label) <= 0.0:
                raise TerrainContractError(f"{label} must be positive")
        _finite(self.item_prior_reference_center, "item_prior_reference_center")
        if not self.response_scale_min < self.response_scale < self.response_scale_max:
            raise TerrainContractError("response scale bounds must contain response_scale")
        if self.item_grid_size < 51 or self.item_grid_size % 2 == 0:
            raise TerrainContractError("item_grid_size must be odd and at least 51")
        if not 0.0 <= self.attempt_information_weight <= 0.5:
            raise TerrainContractError("attempt weight must be in [0, .5]")
        if not 0.0 <= self.composite_weight_exponent <= 1.0:
            raise TerrainContractError("composite weight exponent must be in [0, 1]")
        if len(set(self.style_names)) != len(self.style_names):
            raise TerrainContractError("style_names must be unique")
        if self.coordinate_iterations < 1 or self.coordinate_points < 7:
            raise TerrainContractError("coordinate search settings are too small")
        if self.projection_draws < 200:
            raise TerrainContractError("projection_draws must be at least 200")


@dataclass(frozen=True)
class AthleteTerrainParameters:
    athlete_id: str
    log_scale_ratio: float
    log_scale_sd: float
    procedure_coefficients: tuple[float, float]
    procedure_sds: tuple[float, float]
    pre_zone_style_coefficients: tuple[float, ...]
    pre_zone_style_sds: tuple[float, ...]
    post_zone_style_coefficients: tuple[float, ...]
    post_zone_style_sds: tuple[float, ...]
    item_count: int
    competition_count: int
    mode: str


@dataclass(frozen=True)
class FittedTerrainResponse:
    config: TerrainResponseConfig
    athletes: Mapping[str, AthleteTerrainParameters]
    training_rows: int
    training_competitions: int
    objective: float
    rating_pool_id: str = DEFAULT_RATING_POOL
    warnings: tuple[str, ...] = (
        "leave-one-out item posteriors form a weighted composite likelihood",
        "population procedure/style main effects are absorbed by procedure-conditioned Item-Elo",
        "response fitting integrates the joint Gaussian Zone/Post Item-Elo summary; full item-posterior draws are not retained",
    )

    def __post_init__(self) -> None:
        if not str(self.rating_pool_id).strip():
            raise TerrainContractError("fitted model rating_pool_id is required")


@dataclass(frozen=True)
class FrozenTargetItem:
    problem_id: str
    zone_mean: float
    zone_sd: float
    post_zone_mean: float
    post_zone_sd: float
    zone_post_correlation: float
    attempt_cap: int
    as_of: datetime
    provenance: str
    semantics: str = ITEM_ELO_SEMANTICS
    derived_from_target_results: bool = False
    rating_pool_id: str = DEFAULT_RATING_POOL
    reference_response_scale: float = 155.0
    pre_zone_style_features: Mapping[str, float] = field(default_factory=dict)
    post_zone_style_features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for label in ("zone_mean", "post_zone_mean"):
            _finite(getattr(self, label), label)
        for label in ("zone_sd", "post_zone_sd"):
            if _finite(getattr(self, label), label) < 0.0:
                raise TerrainContractError(f"{label} cannot be negative")
        if not -0.95 <= self.zone_post_correlation <= 0.95:
            raise TerrainContractError("target item correlation must be in [-.95, .95]")
        if self.attempt_cap < 1:
            raise TerrainContractError("attempt_cap must be positive")
        if self.semantics != ITEM_ELO_SEMANTICS:
            raise TerrainContractError("target item has incompatible Elo semantics")
        if not str(self.rating_pool_id).strip():
            raise TerrainContractError("target item rating_pool_id is required")
        if _finite(self.reference_response_scale, "reference_response_scale") <= 0.0:
            raise TerrainContractError("reference_response_scale must be positive")
        source = str(self.provenance).strip().casefold()
        if (
            not source
            or self.derived_from_target_results
            or any(token in source for token in _FORBIDDEN_TARGET_PROVENANCE)
        ):
            raise TerrainContractError("target item distribution cannot use target results")
        _utc(self.as_of, "target item as_of")
        for segment, features in (
            ("pre-zone", self.pre_zone_style_features),
            ("post-zone", self.post_zone_style_features),
        ):
            for name, value in features.items():
                if not str(name).strip():
                    raise TerrainContractError("target style names cannot be empty")
                _finite(value, f"target {segment} style {name}")


@dataclass(frozen=True)
class FrozenTerrainTarget:
    target_id: str
    competition_start: datetime
    frozen_at: datetime
    procedure: str
    items: tuple[FrozenTargetItem, ...]
    projection_estimand: str = "endpoint_vector_only"
    rating_pool_id: str = DEFAULT_RATING_POOL

    def __post_init__(self) -> None:
        start = _utc(self.competition_start, "target competition_start")
        frozen = _utc(self.frozen_at, "target frozen_at")
        if frozen >= start:
            raise TerrainContractError("target must be frozen before competition_start")
        if _procedure(self.procedure) not in CONFIRMED_PROCEDURES:
            raise TerrainContractError("target procedure must be confirmed")
        if not self.target_id or not self.items:
            raise TerrainContractError("target_id and at least one item are required")
        if not str(self.rating_pool_id).strip():
            raise TerrainContractError("target rating_pool_id is required")
        if self.projection_estimand not in PROJECTION_ESTIMANDS:
            raise TerrainContractError("target projection_estimand is not declared")
        reference_scales = {float(item.reference_response_scale) for item in self.items}
        if len(reference_scales) != 1:
            raise TerrainContractError("target items use different reference response scales")
        for item in self.items:
            if item.rating_pool_id != self.rating_pool_id:
                raise TerrainContractError("target item uses a different rating pool")
            if _utc(item.as_of, "target item as_of") > frozen:
                raise TerrainContractError("target item distribution was not available at frozen_at")

    @property
    def confirmed_procedure(self) -> str:
        return _procedure(self.procedure)


@dataclass(frozen=True)
class TerrainProjection:
    athlete_id: str
    projection_elo: float | None
    projection_elo_sd: float | None
    projection_elo_lower: float | None
    projection_elo_upper: float | None
    zone_projection_elo: float
    zone_projection_elo_sd: float
    zone_projection_elo_lower: float
    zone_projection_elo_upper: float
    top_projection_elo: float
    top_projection_elo_sd: float
    top_projection_elo_lower: float
    top_projection_elo_upper: float
    expected_zones: float
    expected_zones_sd: float
    expected_tops: float
    expected_tops_sd: float
    target_id: str
    projection_model: str
    projection_context: str
    projection_estimand: str
    athlete_mode: str
    rating_pool_id: str
    item_elo_semantics: str
    reference_response_scale: float
    warnings: tuple[str, ...]


def _default_prior(
    config: TerrainResponseConfig,
) -> tuple[float, float, float, float, float]:
    # One predeclared population-scale anchor is shared by every held-athlete
    # posterior. Centering this prior on each LOO peer subset makes the prior
    # itself depend on who was held out. A context-specific prior must instead
    # arrive as a FrozenItemPrior dated before the competition.
    center = float(config.item_prior_reference_center)
    return (
        center + config.zone_prior_offset,
        config.item_prior_sd,
        center + config.top_prior_offset,
        config.item_prior_sd,
        0.35,
    )


def _cumulative_top_q50(
    zone_difficulty: np.ndarray,
    post_zone_difficulty: np.ndarray,
    response_scale: float,
) -> np.ndarray:
    """Solve P(zone|q) P(post-zone|zone,q) = .5 elementwise."""

    zone = np.asarray(zone_difficulty, dtype=float)
    post = np.asarray(post_zone_difficulty, dtype=float)
    low = np.minimum(zone, post) - 12.0 * response_scale
    high = np.maximum(zone, post) + 12.0 * response_scale
    for _ in range(70):
        midpoint = (low + high) / 2.0
        probability = _sigmoid((midpoint - zone) / response_scale) * _sigmoid(
            (midpoint - post) / response_scale
        )
        below = probability < 0.5
        low = np.where(below, midpoint, low)
        high = np.where(below, high, midpoint)
    return (low + high) / 2.0


def _joint_item_posterior(
    peers: Sequence[ProblemOutcome],
    prior_values: tuple[float, float, float, float, float],
    config: TerrainResponseConfig,
) -> tuple[HurdleItemElo, HurdleItemElo, HurdleItemElo, float]:
    zone_mean, zone_sd, post_mean, post_sd, correlation = prior_values
    stable_means = [row.stable_mean for row in peers]
    lower = min(stable_means + [zone_mean, post_mean]) - 4.5 * max(
        [config.response_scale, zone_sd, post_sd]
        + [row.stable_sd for row in peers]
    )
    upper = max(stable_means + [zone_mean, post_mean]) + 4.5 * max(
        [config.response_scale, zone_sd, post_sd]
        + [row.stable_sd for row in peers]
    )
    grid = np.linspace(lower, upper, config.item_grid_size)
    z_standard = (grid - zone_mean) / zone_sd
    p_standard = (grid - post_mean) / post_sd
    denominator = 2.0 * (1.0 - correlation**2)
    log_prior = -(
        z_standard[:, None] ** 2
        - 2.0 * correlation * z_standard[:, None] * p_standard[None, :]
        + p_standard[None, :] ** 2
    ) / denominator
    log_prior += -math.log(zone_sd * post_sd * math.sqrt(1.0 - correlation**2))

    log_joint = log_prior.copy()
    zone_successes = 0
    top_successes = 0
    top_opportunities = 0
    for row in peers:
        stage = (
            "top"
            if row.reached_top
            else "zone_only"
            if row.reached_zone
            else "no_zone"
        )
        stage_probability = _ordered_stage_probability_grid(
            row.stable_mean,
            row.stable_sd,
            grid,
            grid,
            config.response_scale,
            stage,
        )
        log_joint += np.log(np.clip(stage_probability, _EPS, 1.0))
        zone_successes += int(row.reached_zone)
        if row.reached_zone:
            top_opportunities += 1
            top_successes += int(row.reached_top)

    log_joint -= _logsumexp(log_joint)
    weights = np.exp(log_joint)
    weights /= weights.sum()
    zone_weights = weights.sum(axis=1)
    post_weights = weights.sum(axis=0)
    z_mean = float(np.sum(grid * zone_weights))
    p_mean = float(np.sum(grid * post_weights))
    z_var = float(np.sum((grid - z_mean) ** 2 * zone_weights))
    p_var = float(np.sum((grid - p_mean) ** 2 * post_weights))
    covariance = float(
        np.sum(
            (grid[:, None] - z_mean)
            * (grid[None, :] - p_mean)
            * weights
        )
    )
    alpha = (1.0 - config.credible_mass) / 2.0

    def status(successes: int, opportunities: int, hurdle: str) -> str:
        if opportunities == 0:
            return f"prior_only_no_peer_{hurdle}_opportunity"
        if successes == 0:
            return f"lower_bound_only_no_peer_{hurdle}_success"
        if successes == opportunities:
            return f"upper_bound_only_all_peer_{hurdle}_success"
        return "mixed_peer_outcomes_identified"

    zone = HurdleItemElo(
        z_mean,
        math.sqrt(max(z_var, 0.0)),
        _weighted_quantile(grid, zone_weights, alpha),
        _weighted_quantile(grid, zone_weights, 1.0 - alpha),
        status(zone_successes, len(peers), "zone"),
        zone_successes,
        len(peers),
    )
    post_zone = HurdleItemElo(
        p_mean,
        math.sqrt(max(p_var, 0.0)),
        _weighted_quantile(grid, post_weights, alpha),
        _weighted_quantile(grid, post_weights, 1.0 - alpha),
        status(top_successes, top_opportunities, "post_zone"),
        top_successes,
        top_opportunities,
    )
    cumulative = _cumulative_top_q50(
        grid[:, None], grid[None, :], config.response_scale
    )
    flat_cumulative = cumulative.ravel()
    flat_weights = weights.ravel()
    cumulative_mean = float(np.sum(flat_cumulative * flat_weights))
    cumulative_variance = float(
        np.sum((flat_cumulative - cumulative_mean) ** 2 * flat_weights)
    )
    top = HurdleItemElo(
        cumulative_mean,
        math.sqrt(max(cumulative_variance, 0.0)),
        _weighted_quantile(flat_cumulative, flat_weights, alpha),
        _weighted_quantile(flat_cumulative, flat_weights, 1.0 - alpha),
        status(top_successes, len(peers), "top"),
        top_successes,
        len(peers),
    )
    return zone, post_zone, top, covariance


def calibrate_leave_one_out_item_elos(
    outcomes: Sequence[ProblemOutcome],
    config: TerrainResponseConfig,
    priors: Mapping[tuple[str, str], FrozenItemPrior] | None = None,
) -> Mapping[tuple[str, str, str], LeaveOneOutItemElo]:
    """Calibrate zone, post-zone and cumulative-top Item-Elo with LOO peers.

    The returned key is ``(competition_id, problem_id, athlete_id)``.  There is
    deliberately no fallback to a full-field posterior when only one peer (or
    no peer) remains.
    """

    grouped: dict[tuple[str, str], list[ProblemOutcome]] = {}
    for row in outcomes:
        grouped.setdefault((row.competition_id, row.problem_id), []).append(row)
    output: dict[tuple[str, str, str], LeaveOneOutItemElo] = {}
    supplied = priors or {}
    for key, rows in grouped.items():
        procedures = {row.confirmed_procedure for row in rows}
        attempt_caps = {row.attempt_cap for row in rows}
        starts = {_utc(row.competition_start, "competition_start") for row in rows}
        rating_pools = {row.rating_pool_id for row in rows}
        if (
            len(procedures) != 1
            or len(starts) != 1
            or len(attempt_caps) != 1
            or len(rating_pools) != 1
        ):
            raise TerrainContractError(
                "one boulder cannot mix procedures, attempt horizons, competition starts "
                "or rating pools"
            )
        rating_pool_id = next(iter(rating_pools))
        prior = supplied.get(key)
        if prior is not None:
            if (prior.competition_id, prior.problem_id) != key:
                raise TerrainContractError("item prior key mismatch")
            if _utc(prior.competition_start, "item prior competition_start") not in starts:
                raise TerrainContractError("item prior competition boundary mismatch")
            if prior.rating_pool_id != rating_pool_id:
                raise TerrainContractError("item prior rating-pool mismatch")
            if not math.isclose(
                prior.reference_response_scale,
                config.response_scale,
                rel_tol=0.0,
                abs_tol=1.0e-9,
            ):
                raise TerrainContractError("item prior reference-response-scale mismatch")
        athlete_ids = [row.athlete_id for row in rows]
        if len(set(athlete_ids)) != len(athlete_ids):
            raise TerrainContractError("one athlete can have only one row per boulder")
        for athlete_id in athlete_ids:
            peers = [row for row in rows if row.athlete_id != athlete_id]
            calibrated_at = (
                max(_utc(row.result_available_at, "result_available_at") for row in peers)
                if peers
                else next(iter(starts))
            )
            if prior is None:
                prior_values = _default_prior(config)
            else:
                prior_values = (
                    prior.zone_mean,
                    prior.zone_sd,
                    prior.post_zone_mean,
                    prior.post_zone_sd,
                    prior.correlation,
                )
            zone, post_zone, top, covariance = _joint_item_posterior(
                peers, prior_values, config
            )
            output[(key[0], key[1], athlete_id)] = LeaveOneOutItemElo(
                competition_id=key[0],
                problem_id=key[1],
                procedure=next(iter(procedures)),
                excluded_athlete_id=athlete_id,
                zone=zone,
                post_zone=post_zone,
                top=top,
                zone_post_covariance=covariance,
                peer_count=len(peers),
                calibrated_at=calibrated_at,
                rating_pool_id=rating_pool_id,
                reference_response_scale=config.response_scale,
            )
    return output


def _conditional_geometric_log_probability(attempt: int, hazard: float, cap: int) -> float:
    if attempt < 1 or attempt > cap:
        return math.log(_EPS)
    h = float(np.clip(hazard, _EPS, 1.0 - _EPS))
    normalizer = max(1.0 - (1.0 - h) ** cap, _EPS)
    return float((attempt - 1) * math.log1p(-h) + math.log(h) - math.log(normalizer))


def _style_vector(
    features: Mapping[str, float], config: TerrainResponseConfig
) -> np.ndarray:
    return np.asarray(
        [
            (float(features.get(name, config.style_center)) - config.style_center)
            / config.style_scale
            for name in config.style_names
        ],
        dtype=float,
    )


def _integrated_ordered_response_probabilities(
    row: ProblemOutcome,
    item: LeaveOneOutItemElo,
    response_scale: float,
    zone_shift: float,
    post_shift: float,
    *,
    quadrature_nodes: int = 21,
) -> tuple[float, float, float]:
    """Return P(Zone), P(Top), P(Top|Zone) under one joint latent draw.

    The athlete state is shared by both hurdles and the LOO Zone/Post item
    posterior carries covariance. The available item object exposes Gaussian
    summaries, so this is exact for that declared approximation; retaining
    full posterior draws is a later challenger.
    """

    if quadrature_nodes < 9:
        raise TerrainContractError("response quadrature requires >=9 nodes")
    zone_variance = float(row.stable_sd**2 + item.zone.sd**2)
    post_variance = float(row.stable_sd**2 + item.post_zone.sd**2)
    covariance = float(row.stable_sd**2 + item.zone_post_covariance)
    covariance_limit = math.sqrt(max(zone_variance * post_variance, 0.0))
    covariance = float(np.clip(covariance, -covariance_limit, covariance_limit))
    covariance_matrix = np.asarray(
        [[zone_variance, covariance], [covariance, post_variance]], dtype=float
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    nodes, weights = np.polynomial.hermite.hermgauss(int(quadrature_nodes))
    standard = math.sqrt(2.0) * np.column_stack(
        (np.repeat(nodes, len(nodes)), np.tile(nodes, len(nodes)))
    )
    joint_weights = np.outer(weights, weights).ravel() / math.pi
    means = np.asarray(
        [
            row.stable_mean + zone_shift - item.zone.mean,
            row.stable_mean + post_shift - item.post_zone.mean,
        ],
        dtype=float,
    )
    latent = means[None, :] + standard @ root.T
    p_zone_draw = _sigmoid(latent[:, 0] / response_scale)
    p_post_draw = _sigmoid(latent[:, 1] / response_scale)
    p_zone = float(joint_weights @ p_zone_draw)
    p_top = float(joint_weights @ (p_zone_draw * p_post_draw))
    p_zone = float(np.clip(p_zone, _EPS, 1.0 - _EPS))
    p_top = float(np.clip(p_top, _EPS, max(p_zone - _EPS, _EPS)))
    p_post_given_zone = float(np.clip(p_top / p_zone, _EPS, 1.0 - _EPS))
    return p_zone, p_top, p_post_given_zone


def _row_log_likelihood(
    row: ProblemOutcome,
    item: LeaveOneOutItemElo,
    parameters: np.ndarray,
    config: TerrainResponseConfig,
    attempt_eligible: bool,
) -> float:
    log_ratio = float(parameters[0])
    procedure_coefficients = parameters[1:3]
    style_count = len(config.style_names)
    pre_style_coefficients = parameters[3 : 3 + style_count]
    post_style_coefficients = parameters[3 + style_count :]
    response_scale = float(
        np.clip(
            config.response_scale * math.exp(log_ratio),
            config.response_scale_min,
            config.response_scale_max,
        )
    )
    procedure_shift = float(
        _procedure_codes(row.confirmed_procedure) @ procedure_coefficients
    )
    zone_shift = procedure_shift
    post_shift = procedure_shift
    if style_count:
        zone_shift += float(
            _style_vector(row.pre_zone_style_features, config)
            @ pre_style_coefficients
        )
        post_shift += float(
            _style_vector(row.post_zone_style_features, config)
            @ post_style_coefficients
        )
    cap = row.attempt_cap

    p_zone, p_top, p_post_given_zone = _integrated_ordered_response_probabilities(
        row, item, response_scale, zone_shift, post_shift
    )
    h_zone = 1.0 - (1.0 - p_zone) ** (1.0 / cap)
    if not row.reached_zone:
        value = math.log(max(1.0 - p_zone, _EPS))
    elif row.reached_top:
        value = math.log(max(p_top, _EPS))
    else:
        value = math.log(max(p_zone - p_top, _EPS))
    if row.reached_zone:
        if attempt_eligible and row.attempts_to_zone is not None:
            value += config.attempt_information_weight * _conditional_geometric_log_probability(
                row.attempts_to_zone, h_zone, cap
            )
        if (
            attempt_eligible
            and row.reached_top
            and row.post_zone_opportunities is not None
            and row.post_zone_success_index is not None
        ):
            # Aggregate attempts_to_top cannot reveal how many attempts reached
            # Zone.  Only an explicit, externally reviewed conditional
            # opportunity horizon can refine post-zone efficiency.  The term is
            # conditional on eventual Top, so the stage likelihood above is not
            # counted twice.
            post_cap = int(row.post_zone_opportunities)
            post_hazard = 1.0 - (1.0 - p_post_given_zone) ** (1.0 / post_cap)
            value += config.attempt_information_weight * _conditional_geometric_log_probability(
                int(row.post_zone_success_index), post_hazard, post_cap
            )
    return float(value)


def _coordinate_map(
    rows: Sequence[ProblemOutcome],
    items: Sequence[LeaveOneOutItemElo],
    eligibilities: Sequence[bool],
    weights: Sequence[float],
    config: TerrainResponseConfig,
    active: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    size = 3 + 2 * len(config.style_names)
    parameters = np.zeros(size, dtype=float)
    prior_sds = np.asarray(
        [config.log_scale_prior_sd]
        + [config.athlete_procedure_prior_sd] * 2
        + [config.athlete_style_prior_sd] * (2 * len(config.style_names)),
        dtype=float,
    )
    bounds = [
        (
            math.log(config.response_scale_min / config.response_scale),
            math.log(config.response_scale_max / config.response_scale),
        )
    ] + [(-320.0, 320.0)] * (size - 1)
    if active.shape != (size,):
        raise TerrainContractError("active parameter mask has the wrong shape")

    def objective(values: np.ndarray) -> float:
        penalty = 0.5 * float(np.sum((values / prior_sds) ** 2))
        likelihood = sum(
            weight * _row_log_likelihood(row, item, values, config, eligible)
            for row, item, eligible, weight in zip(rows, items, eligibilities, weights)
        )
        return float(penalty - likelihood)

    spans = np.asarray([0.75] + [180.0] * (size - 1), dtype=float)
    for _ in range(config.coordinate_iterations):
        for index in range(size):
            if not active[index]:
                parameters[index] = 0.0
                continue
            low = max(bounds[index][0], parameters[index] - spans[index])
            high = min(bounds[index][1], parameters[index] + spans[index])
            candidates = np.linspace(low, high, config.coordinate_points)
            scores = []
            for candidate in candidates:
                trial = parameters.copy()
                trial[index] = candidate
                scores.append(objective(trial))
            parameters[index] = float(candidates[int(np.argmin(scores))])
        spans *= 0.45

    sds = np.empty(size, dtype=float)
    center_objective = objective(parameters)
    for index in range(size):
        if not active[index]:
            sds[index] = prior_sds[index]
            continue
        step = max(spans[index], 0.01 if index == 0 else 2.0)
        left = parameters.copy()
        right = parameters.copy()
        left[index] = max(bounds[index][0], parameters[index] - step)
        right[index] = min(bounds[index][1], parameters[index] + step)
        actual_left = parameters[index] - left[index]
        actual_right = right[index] - parameters[index]
        if actual_left <= 0.0 or actual_right <= 0.0:
            sds[index] = prior_sds[index]
            continue
        # Unequal-step second derivative.
        curvature = 2.0 * (
            actual_left * objective(right)
            + actual_right * objective(left)
            - (actual_left + actual_right) * center_objective
        ) / (actual_left * actual_right * (actual_left + actual_right))
        sds[index] = min(prior_sds[index], math.sqrt(1.0 / max(curvature, 1.0e-9)))
    return parameters, sds, center_objective


def fit_terrain_response(
    outcomes: Sequence[ProblemOutcome],
    item_elos: Mapping[tuple[str, str, str], LeaveOneOutItemElo],
    config: TerrainResponseConfig,
) -> FittedTerrainResponse:
    """Fit athlete response deviations on development evidence only."""

    if not outcomes:
        raise TerrainContractError("at least one development outcome is required")
    development_end = _utc(config.development_end, "development_end")
    rating_pools = {row.rating_pool_id for row in outcomes}
    if len(rating_pools) != 1:
        raise TerrainContractError("one terrain-response fit cannot mix rating pools")
    rating_pool_id = next(iter(rating_pools))
    stage_counts: dict[tuple[str, str, str], int] = {}
    field_sizes: dict[tuple[str, str], int] = {}
    frozen_states: dict[tuple[str, str], tuple[datetime, float, float]] = {}
    for row in outcomes:
        if _utc(row.result_available_at, "result_available_at") > development_end:
            raise TerrainContractError("locked/test result entered terrain-response fit")
        key = (row.competition_id, row.problem_id, row.athlete_id)
        if key not in item_elos:
            raise TerrainContractError(f"missing leave-one-out Item-Elo for {key}")
        item = item_elos[key]
        if item.excluded_athlete_id != row.athlete_id:
            raise TerrainContractError("Item-Elo did not exclude the evaluated athlete")
        if (item.competition_id, item.problem_id) != (row.competition_id, row.problem_id):
            raise TerrainContractError("Item-Elo and outcome identify different boulders")
        if item.semantics != ITEM_ELO_SEMANTICS:
            raise TerrainContractError("Item-Elo semantics mismatch")
        if item.rating_pool_id != row.rating_pool_id:
            raise TerrainContractError("Item-Elo and outcome use different rating pools")
        if not math.isclose(
            item.reference_response_scale,
            config.response_scale,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise TerrainContractError("Item-Elo reference-response-scale mismatch")
        if _utc(item.calibrated_at, "Item-Elo calibrated_at") > development_end:
            raise TerrainContractError("locked/test peer result entered Item-Elo calibration")
        state_key = (row.competition_id, row.athlete_id)
        frozen_state = (
            _utc(row.frozen_at, "frozen_at"),
            float(row.stable_mean),
            float(row.stable_sd),
        )
        previous_state = frozen_states.setdefault(state_key, frozen_state)
        if previous_state != frozen_state:
            raise TerrainContractError(
                "athlete stable readiness changed inside one competition"
            )
        stage_counts[(row.competition_id, row.problem_id, row.stage)] = (
            stage_counts.get((row.competition_id, row.problem_id, row.stage), 0) + 1
        )
        field_key = (row.competition_id, row.problem_id)
        field_sizes[field_key] = field_sizes.get(field_key, 0) + 1

    grouped: dict[str, list[ProblemOutcome]] = {}
    for row in outcomes:
        grouped.setdefault(row.athlete_id, []).append(row)
    fitted: dict[str, AthleteTerrainParameters] = {}
    total_objective = 0.0
    for athlete_id, rows in sorted(grouped.items()):
        items = [item_elos[(row.competition_id, row.problem_id, athlete_id)] for row in rows]
        eligibilities = [
            stage_counts[(row.competition_id, row.problem_id, row.stage)] >= 2
            for row in rows
        ]
        weights = [
            1.0
            / field_sizes[(row.competition_id, row.problem_id)]
            ** config.composite_weight_exponent
            for row in rows
        ]
        procedures = {row.confirmed_procedure for row in rows}
        hurdle_difficulties = np.asarray(
            [value for item in items for value in (item.zone.mean, item.post_zone.mean)],
            dtype=float,
        )
        style_count = len(config.style_names)
        active = np.ones(3 + 2 * style_count, dtype=bool)
        active[0] = (
            len(hurdle_difficulties) >= 2
            and float(np.ptp(hurdle_difficulties)) >= config.minimum_difficulty_spread
        )
        # An athlete-only procedure effect is a within-athlete contrast.  With
        # one procedure it is indistinguishable from a free intercept and is
        # therefore held at the zero-centred population reference.
        active[1:3] = len(procedures) >= 2
        for offset, style_name in enumerate(config.style_names):
            pre_values = np.asarray(
                [
                    row.pre_zone_style_features.get(style_name, config.style_center)
                    for row in rows
                ],
                dtype=float,
            )
            active[3 + offset] = (
                len(pre_values) >= 2
                and float(np.ptp(pre_values)) >= config.minimum_style_spread
            )
            post_values = np.asarray(
                [
                    row.post_zone_style_features.get(style_name, config.style_center)
                    for row in rows
                    if row.reached_zone
                ],
                dtype=float,
            )
            active[3 + style_count + offset] = (
                len(post_values) >= 2
                and float(np.ptp(post_values)) >= config.minimum_style_spread
            )
        parameters, sds, objective = _coordinate_map(
            rows, items, eligibilities, weights, config, active
        )
        total_objective += objective
        count = len(rows)
        fitted[athlete_id] = AthleteTerrainParameters(
            athlete_id=athlete_id,
            log_scale_ratio=float(parameters[0]),
            log_scale_sd=float(sds[0]),
            procedure_coefficients=(float(parameters[1]), float(parameters[2])),
            procedure_sds=(float(sds[1]), float(sds[2])),
            pre_zone_style_coefficients=tuple(
                float(value) for value in parameters[3 : 3 + style_count]
            ),
            pre_zone_style_sds=tuple(
                float(value) for value in sds[3 : 3 + style_count]
            ),
            post_zone_style_coefficients=tuple(
                float(value) for value in parameters[3 + style_count :]
            ),
            post_zone_style_sds=tuple(
                float(value) for value in sds[3 + style_count :]
            ),
            item_count=count,
            competition_count=len({row.competition_id for row in rows}),
            mode=(
                "partially_pooled_athlete"
                if count >= config.minimum_athlete_items
                else "strongly_pooled_sparse_athlete"
            ),
        )
    return FittedTerrainResponse(
        config=config,
        athletes=fitted,
        training_rows=len(outcomes),
        training_competitions=len({row.competition_id for row in outcomes}),
        objective=float(total_objective),
        rating_pool_id=rating_pool_id,
    )


def hurdle_success_probability(
    stable_elo: float,
    hurdle_item_elo: np.ndarray | float,
    response_scale: float,
    *,
    shift: float = 0.0,
) -> np.ndarray | float:
    """Event-level hurdle probability; Item-Elo is exactly its q50."""

    if response_scale <= 0.0:
        raise TerrainContractError("response_scale must be positive")
    return _sigmoid(
        (float(stable_elo) + float(shift) - np.asarray(hurdle_item_elo))
        / response_scale
    )


def project_terrain_target(
    athlete_id: str,
    stable_mean: float,
    stable_sd: float,
    target: FrozenTerrainTarget,
    fitted: FittedTerrainResponse,
) -> TerrainProjection:
    """Integrate a frozen target difficulty distribution into Projection-Elo."""

    config = fitted.config
    if stable_sd <= 0.0:
        raise TerrainContractError("stable_sd must be positive")
    if target.rating_pool_id != fitted.rating_pool_id:
        raise TerrainContractError("target and fitted model use different rating pools")
    for item in target.items:
        if not math.isclose(
            item.reference_response_scale,
            config.response_scale,
            rel_tol=0.0,
            abs_tol=1.0e-9,
        ):
            raise TerrainContractError("target item reference-response-scale mismatch")
    missing_pre_styles = sorted(
        {
            name
            for item in target.items
            for name in config.style_names
            if name not in item.pre_zone_style_features
        }
    )
    missing_post_styles = sorted(
        {
            name
            for item in target.items
            for name in config.style_names
            if name not in item.post_zone_style_features
        }
    )
    parameters = fitted.athletes.get(athlete_id)
    if parameters is None:
        parameters = AthleteTerrainParameters(
            athlete_id,
            0.0,
            config.log_scale_prior_sd,
            (0.0, 0.0),
            (config.athlete_procedure_prior_sd,) * 2,
            (0.0,) * len(config.style_names),
            (config.athlete_style_prior_sd,) * len(config.style_names),
            (0.0,) * len(config.style_names),
            (config.athlete_style_prior_sd,) * len(config.style_names),
            0,
            0,
            "population_reference_unseen_athlete",
        )
    rng = np.random.default_rng(config.projection_seed)
    draws = config.projection_draws
    stable = rng.normal(float(stable_mean), float(stable_sd), draws)
    log_scale = np.clip(
        rng.normal(parameters.log_scale_ratio, parameters.log_scale_sd, draws),
        math.log(config.response_scale_min / config.response_scale),
        math.log(config.response_scale_max / config.response_scale),
    )
    response_scale = config.response_scale * np.exp(log_scale)
    procedure_coefficients = np.column_stack(
        [
            rng.normal(parameters.procedure_coefficients[index], parameters.procedure_sds[index], draws)
            for index in range(2)
        ]
    )
    procedure_shift = procedure_coefficients @ _procedure_codes(
        target.confirmed_procedure
    )
    pre_style_coefficients = np.empty((draws, 0), dtype=float)
    post_style_coefficients = np.empty((draws, 0), dtype=float)
    if config.style_names:
        pre_style_coefficients = np.column_stack(
            [
                rng.normal(
                    parameters.pre_zone_style_coefficients[index],
                    parameters.pre_zone_style_sds[index],
                    draws,
                )
                for index in range(len(config.style_names))
            ]
        )
        post_style_coefficients = np.column_stack(
            [
                rng.normal(
                    parameters.post_zone_style_coefficients[index],
                    parameters.post_zone_style_sds[index],
                    draws,
                )
                for index in range(len(config.style_names))
            ]
        )

    expected_zones = np.zeros(draws, dtype=float)
    expected_tops = np.zeros(draws, dtype=float)
    item_draws: list[tuple[np.ndarray, np.ndarray]] = []
    for item in target.items:
        covariance = np.asarray(
            [
                [
                    item.zone_sd**2,
                    item.zone_post_correlation * item.zone_sd * item.post_zone_sd,
                ],
                [
                    item.zone_post_correlation * item.zone_sd * item.post_zone_sd,
                    item.post_zone_sd**2,
                ],
            ],
            dtype=float,
        )
        if item.zone_sd == 0.0 and item.post_zone_sd == 0.0:
            item_sample = np.tile(
                [item.zone_mean, item.post_zone_mean], (draws, 1)
            )
        else:
            covariance += np.eye(2) * 1.0e-9
            item_sample = rng.multivariate_normal(
                [item.zone_mean, item.post_zone_mean], covariance, size=draws
            )
        zone_difficulty = item_sample[:, 0]
        post_difficulty = item_sample[:, 1]
        zone_shift = procedure_shift.copy()
        post_shift = procedure_shift.copy()
        if config.style_names:
            zone_shift += pre_style_coefficients @ _style_vector(
                item.pre_zone_style_features, config
            )
            post_shift += post_style_coefficients @ _style_vector(
                item.post_zone_style_features, config
            )
        p_zone = _sigmoid(
            (stable + zone_shift - zone_difficulty) / response_scale
        )
        p_post_conditional = _sigmoid(
            (stable + post_shift - post_difficulty) / response_scale
        )
        expected_zones += p_zone
        expected_tops += p_zone * p_post_conditional
        item_draws.append((zone_difficulty, post_difficulty))

    low = np.full(
        draws,
        min(
            min(item.zone_mean, item.post_zone_mean) for item in target.items
        )
        - 3000.0,
    )
    high = np.full(
        draws,
        max(
            max(item.zone_mean, item.post_zone_mean) for item in target.items
        )
        + 3000.0,
    )

    def solve_equivalent(
        target_values: np.ndarray, zone_weight: float, top_weight: float
    ) -> np.ndarray:
        local_low = low.copy()
        local_high = high.copy()
        # The same sampled target terrain appears on both sides.  The fixed
        # population response curve defines the reported Elo scale when the
        # athlete-specific response slope differs.
        for _ in range(55):
            midpoint = (local_low + local_high) / 2.0
            reference_zones = np.zeros(draws, dtype=float)
            reference_tops = np.zeros(draws, dtype=float)
            for zone_difficulty, post_difficulty in item_draws:
                p_zone = _sigmoid(
                    (midpoint - zone_difficulty) / config.response_scale
                )
                p_post = _sigmoid(
                    (midpoint - post_difficulty) / config.response_scale
                )
                reference_zones += p_zone
                reference_tops += p_zone * p_post
            reference_value = (
                zone_weight * reference_zones + top_weight * reference_tops
            )
            below = reference_value < target_values
            local_low[below] = midpoint[below]
            local_high[~below] = midpoint[~below]
        return (local_low + local_high) / 2.0

    zone_equivalent = solve_equivalent(expected_zones, 1.0, 0.0)
    top_equivalent = solve_equivalent(expected_tops, 0.0, 1.0)
    overall_equivalent: np.ndarray | None = None
    if (
        target.projection_estimand
        == "modern_25_10_achievement_without_attempt_penalty"
    ):
        modern_achievement = 10.0 * expected_zones + 15.0 * expected_tops
        overall_equivalent = solve_equivalent(modern_achievement, 10.0, 15.0)

    alpha = (1.0 - config.credible_mass) / 2.0
    warnings = list(fitted.warnings)
    if missing_pre_styles:
        warnings.append(
            "missing pre-zone target style tags pooled to neutral: "
            + ", ".join(missing_pre_styles)
        )
    if missing_post_styles:
        warnings.append(
            "missing post-zone target style tags pooled to neutral: "
            + ", ".join(missing_post_styles)
        )
    if parameters.item_count < config.minimum_athlete_items:
        warnings.append("athlete terrain response remains strongly pooled")

    def summary(values: np.ndarray) -> tuple[float, float, float, float]:
        return (
            float(np.mean(values)),
            float(np.std(values, ddof=1)),
            float(np.quantile(values, alpha)),
            float(np.quantile(values, 1.0 - alpha)),
        )

    zone_summary = summary(zone_equivalent)
    top_summary = summary(top_equivalent)
    overall_summary = summary(overall_equivalent) if overall_equivalent is not None else None
    return TerrainProjection(
        athlete_id=athlete_id,
        projection_elo=overall_summary[0] if overall_summary else None,
        projection_elo_sd=overall_summary[1] if overall_summary else None,
        projection_elo_lower=overall_summary[2] if overall_summary else None,
        projection_elo_upper=overall_summary[3] if overall_summary else None,
        zone_projection_elo=zone_summary[0],
        zone_projection_elo_sd=zone_summary[1],
        zone_projection_elo_lower=zone_summary[2],
        zone_projection_elo_upper=zone_summary[3],
        top_projection_elo=top_summary[0],
        top_projection_elo_sd=top_summary[1],
        top_projection_elo_lower=top_summary[2],
        top_projection_elo_upper=top_summary[3],
        expected_zones=float(np.mean(expected_zones)),
        expected_zones_sd=float(np.std(expected_zones, ddof=1)),
        expected_tops=float(np.mean(expected_tops)),
        expected_tops_sd=float(np.std(expected_tops, ddof=1)),
        target_id=target.target_id,
        projection_model=MODEL_NAME,
        projection_context=(
            f"{target.confirmed_procedure}; {len(target.items)} frozen procedure-conditioned items"
        ),
        projection_estimand=target.projection_estimand,
        athlete_mode=parameters.mode,
        rating_pool_id=target.rating_pool_id,
        item_elo_semantics=ITEM_ELO_SEMANTICS,
        reference_response_scale=config.response_scale,
        warnings=tuple(dict.fromkeys(warnings)),
    )


__all__ = [
    "AthleteTerrainParameters",
    "CONFIRMED_PROCEDURES",
    "DEFAULT_RATING_POOL",
    "FittedTerrainResponse",
    "FrozenItemPrior",
    "FrozenTargetItem",
    "FrozenTerrainTarget",
    "HurdleItemElo",
    "ITEM_ELO_SEMANTICS",
    "LeaveOneOutItemElo",
    "MODEL_NAME",
    "PROJECTION_ESTIMANDS",
    "ProblemOutcome",
    "TerrainContractError",
    "TerrainProjection",
    "TerrainResponseConfig",
    "calibrate_leave_one_out_item_elos",
    "fit_terrain_response",
    "hurdle_success_probability",
    "project_terrain_target",
]
