"""Research-only dynamic Boulder readiness challenger.

This module deliberately does **not** replace the production rating replay.  It
implements the smallest Stage-A challenger that can test the model contract in
``docs/DYNAMIC_BOULDER_READINESS_MODEL_V5.md`` without adding a heavy Bayesian
dependency:

* one broad Gaussian state per athlete;
* a slowly changing readiness component and a separate, mean-reverting form;
* shrunk target offsets for WC+, Canada and other competition domains;
* one frozen pre-event state for every participant;
* an event-balanced Bradley--Terry pairwise composite likelihood over the full
  ranking (explicitly **not** a full Plackett--Luce likelihood);
* provisional quarantine: newcomers learn from established athletes but cannot
  move those athletes until promotion; and
* explicit mean, uncertainty, information and promotion outputs.

The implementation is an assumed-density / Laplace approximation with a full
within-athlete covariance matrix for stable skill, form and target offsets.  It
is intentionally pure NumPy, deterministic and small enough for free hosting.
Chronological locked testing must beat production before any public rating may
use it.  A later Stage-B model should replace the ranking likelihood with the
problem-level hurdle/item-response model once boulder evidence is sufficiently
complete.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


LOG_10 = math.log(10.0)
UPDATE_LINK_CONTRACT = "base10_logistic_normal_attenuation_v1"


def _finite(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return float(1.0 / (1.0 + math.exp(-min(value, 60.0))))
    exponent = math.exp(max(value, -60.0))
    return float(exponent / (1.0 + exponent))


@dataclass(frozen=True)
class DynamicRatingConfig:
    """Predeclared Stage-A assumptions.

    Defaults are challenger priors, not fitted sport-science findings.  They
    must be selected on a development period and frozen before validation.
    """

    display_scale: float = 400.0
    prior_mean: float = 1800.0
    prior_skill_sd: float = 480.0
    minimum_skill_sd: float = 45.0
    minimum_form_sd: float = 20.0
    minimum_target_offset_sd: float = 15.0
    maximum_skill_sd: float = 650.0
    skill_drift_sd_per_year: float = 75.0
    form_stationary_sd: float = 125.0
    form_half_life_days: float = 100.0
    enable_form: bool = True
    offset_prior_sd: float = 110.0
    offset_half_life_days: float = 730.0
    enable_target_offsets: bool = True
    event_performance_sd: float = 155.0
    # This is an asymptotic within-competition information budget.  The
    # realised cap rises logarithmically with field size, reaching the budget
    # at ``field_information_reference_opponents``.  Thus a 1/80 result carries
    # more information than a 1/8 result, but not eleven times as much.
    event_pair_weight_cap: float = 6.0
    field_information_reference_opponents: int = 79
    field_information_floor_fraction: float = 0.50
    provisional_relative_shift_cap: float = 180.0
    score_gap_weight: float = 0.45
    score_gap_scale: float = 0.25
    # Optional robust-innovation challenger.  ``None`` is the default because
    # an arbitrary movement cap must not replace the probabilistic likelihood.
    # Locked ablations may declare a finite cap before evaluation.
    maximum_projection_shift: float | None = None
    promotion_min_anchored_events: int = 3
    promotion_min_anchored_comparisons: int = 12
    promotion_min_unique_opponents: int = 6
    promotion_min_effective_weight: float = 10.0
    promotion_max_skill_sd: float = 220.0
    target_domains: tuple[str, ...] = (
        "wc+",
        "ifsc_non_wc",
        "canada_senior_national",
        "canada_youth_national",
        "nacs",
        "provincial_local",
        "other",
    )
    reference_domain: str = "other"

    def __post_init__(self) -> None:
        positive = {
            "display_scale": self.display_scale,
            "prior_skill_sd": self.prior_skill_sd,
            "minimum_skill_sd": self.minimum_skill_sd,
            "minimum_form_sd": self.minimum_form_sd,
            "minimum_target_offset_sd": self.minimum_target_offset_sd,
            "maximum_skill_sd": self.maximum_skill_sd,
            "form_stationary_sd": self.form_stationary_sd,
            "form_half_life_days": self.form_half_life_days,
            "offset_prior_sd": self.offset_prior_sd,
            "offset_half_life_days": self.offset_half_life_days,
            "event_performance_sd": self.event_performance_sd,
            "event_pair_weight_cap": self.event_pair_weight_cap,
            "provisional_relative_shift_cap": self.provisional_relative_shift_cap,
            "score_gap_scale": self.score_gap_scale,
        }
        for label, value in positive.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{label} must be positive and finite")
        if self.skill_drift_sd_per_year < 0.0:
            raise ValueError("skill_drift_sd_per_year cannot be negative")
        if self.maximum_projection_shift is not None and (
            not math.isfinite(float(self.maximum_projection_shift))
            or float(self.maximum_projection_shift) <= 0.0
        ):
            raise ValueError("maximum_projection_shift must be positive or None")
        if not 0.0 <= self.score_gap_weight <= 2.0:
            raise ValueError("score_gap_weight must be between zero and two")
        if self.minimum_skill_sd >= self.maximum_skill_sd:
            raise ValueError("minimum_skill_sd must be below maximum_skill_sd")
        if self.minimum_form_sd > self.form_stationary_sd:
            raise ValueError("minimum_form_sd cannot exceed form_stationary_sd")
        if self.minimum_target_offset_sd > self.offset_prior_sd:
            raise ValueError(
                "minimum_target_offset_sd cannot exceed offset_prior_sd"
            )
        if self.field_information_reference_opponents < 2:
            raise ValueError(
                "field_information_reference_opponents must be at least two"
            )
        if not 0.0 < self.field_information_floor_fraction <= 1.0:
            raise ValueError(
                "field_information_floor_fraction must be in (0, 1]"
            )
        if self.promotion_min_anchored_events < 1:
            raise ValueError("promotion_min_anchored_events must be positive")
        if self.promotion_min_anchored_comparisons < 1:
            raise ValueError("promotion_min_anchored_comparisons must be positive")
        if self.promotion_min_unique_opponents < 1:
            raise ValueError("promotion_min_unique_opponents must be positive")
        if self.promotion_min_effective_weight <= 0.0:
            raise ValueError("promotion_min_effective_weight must be positive")
        domains = tuple(str(item).strip().lower() for item in self.target_domains)
        if len(domains) != len(set(domains)) or "other" not in domains:
            raise ValueError("target_domains must be unique and include 'other'")
        if str(self.reference_domain).strip().lower() not in domains:
            raise ValueError("reference_domain must be one of target_domains")


@dataclass
class AthleteState:
    """Posterior state after the athlete's latest processed competition."""

    athlete_id: str
    skill_mean: float
    skill_variance: float
    form_mean: float
    form_variance: float
    target_offset_mean: dict[str, float]
    target_offset_variance: dict[str, float]
    # Off-diagonal posterior covariances keyed by sorted component labels.
    # Marginal variances remain in the public fields above for compatibility
    # with audit code and transparent state exports.
    component_covariance: dict[tuple[str, str], float]
    provisional: bool = True
    events_seen: int = 0
    anchored_events: int = 0
    anchored_comparisons: int = 0
    anchored_event_ids: set[str] = field(default_factory=set)
    anchored_opponent_ids: set[str] = field(default_factory=set)
    anchored_effective_weight: float = 0.0
    last_event_time: float | None = None

    def clone(self) -> "AthleteState":
        return AthleteState(
            athlete_id=self.athlete_id,
            skill_mean=float(self.skill_mean),
            skill_variance=float(self.skill_variance),
            form_mean=float(self.form_mean),
            form_variance=float(self.form_variance),
            target_offset_mean=dict(self.target_offset_mean),
            target_offset_variance=dict(self.target_offset_variance),
            component_covariance=dict(self.component_covariance),
            provisional=bool(self.provisional),
            events_seen=int(self.events_seen),
            anchored_events=int(self.anchored_events),
            anchored_comparisons=int(self.anchored_comparisons),
            anchored_event_ids=set(self.anchored_event_ids),
            anchored_opponent_ids=set(self.anchored_opponent_ids),
            anchored_effective_weight=float(self.anchored_effective_weight),
            last_event_time=self.last_event_time,
        )


@dataclass(frozen=True)
class RankedObservation:
    """One athlete's outcome in one jointly processed competition."""

    athlete_id: str
    rank: float
    score_signal: float | None = None


@dataclass(frozen=True)
class RankedContest:
    """One comparable ranking set inside a jointly frozen competition."""

    contest_id: str
    observations: tuple[RankedObservation, ...]
    evidence_group: str | None = None


@dataclass(frozen=True)
class Projection:
    """A dated target prediction before on-day performance noise is sampled."""

    athlete_id: str
    target_domain: str
    event_time: float
    mean: float
    rating_sd: float
    predictive_sd: float
    skill_mean: float
    form_adjustment: float
    target_adjustment: float
    provisional: bool
    anchored_events: int
    anchored_comparisons: int
    unique_anchored_opponents: int
    anchored_effective_weight: float


@dataclass(frozen=True)
class AthleteEventUpdate:
    """Traceable evidence and posterior movement from one frozen event."""

    athlete_id: str
    rank: float
    contest_count: int
    provisional_before: bool
    provisional_after: bool
    pre_projection_mean: float
    pre_rating_sd: float
    observed_pair_score: float | None
    expected_pair_score: float | None
    performance_evidence_mean: float | None
    performance_evidence_sd: float | None
    eligible_established_opponents: int
    eligible_provisional_opponents: int
    effective_pair_weight: float
    relative_pair_weight: float
    score_weight_used: bool
    projection_shift: float
    post_skill_mean: float
    post_form_adjustment: float
    post_target_adjustment: float
    post_projection_mean: float
    post_rating_sd: float
    anchored_events: int
    anchored_comparisons: int
    unique_anchored_opponents: int
    anchored_effective_weight: float
    status: str


@dataclass(frozen=True)
class EventUpdate:
    """All updates computed from the same immutable pre-event state."""

    event_id: str
    event_time: float
    target_domain: str
    participant_count: int
    contest_count: int
    established_count_before: int
    updates: tuple[AthleteEventUpdate, ...]


class DynamicBoulderRating:
    """Pure-NumPy chronological Stage-A rating challenger."""

    def __init__(self, config: DynamicRatingConfig | None = None) -> None:
        self.config = config or DynamicRatingConfig()
        self.states: dict[str, AthleteState] = {}
        self.history: list[EventUpdate] = []
        self._processed_events: set[str] = set()
        self._last_event_time: float | None = None

    def _domain(self, value: object) -> str:
        domain = str(value).strip().lower()
        aliases = {
            "canada": "canada_senior_national",
            "canadian nationals": "canada_senior_national",
            "youth nationals": "canada_youth_national",
            "world": "wc+",
            "ifsc": "ifsc_non_wc",
            "local": "provincial_local",
            "provincial": "provincial_local",
        }
        domain = aliases.get(domain, domain)
        return domain if domain in self.config.target_domains else "other"

    @staticmethod
    def _offset_label(domain: str) -> str:
        return f"offset:{domain}"

    def _component_labels(self) -> tuple[str, ...]:
        labels = ["skill"]
        if self.config.enable_form:
            labels.append("form")
        if self.config.enable_target_offsets:
            labels.extend(
                self._offset_label(domain)
                for domain in self.config.target_domains
                if domain != self.config.reference_domain
            )
        return tuple(labels)

    @staticmethod
    def _covariance_key(left: str, right: str) -> tuple[str, str]:
        return tuple(sorted((left, right)))  # type: ignore[return-value]

    def _component_floor_variance(self, label: str) -> float:
        cfg = self.config
        if label == "skill":
            return float(cfg.minimum_skill_sd**2)
        if label == "form":
            return float(cfg.minimum_form_sd**2)
        return float(cfg.minimum_target_offset_sd**2)

    def _covariance_matrix(
        self, state: AthleteState
    ) -> tuple[tuple[str, ...], np.ndarray]:
        """Return the coherent within-athlete covariance matrix.

        Public marginal fields are treated as canonical so an audit may still
        perturb ``skill_variance`` directly.  Off-diagonal terms are stored in
        ``component_covariance`` and are never silently discarded.
        """

        labels = self._component_labels()
        matrix = np.zeros((len(labels), len(labels)), dtype=float)
        for index, label in enumerate(labels):
            if label == "skill":
                variance = state.skill_variance
            elif label == "form":
                variance = state.form_variance
            else:
                variance = state.target_offset_variance[label.split(":", 1)[1]]
            matrix[index, index] = max(float(variance), 0.0)
        for left_index, left in enumerate(labels):
            for right_index in range(left_index + 1, len(labels)):
                right = labels[right_index]
                covariance = state.component_covariance.get(
                    self._covariance_key(left, right), 0.0
                )
                matrix[left_index, right_index] = float(covariance)
                matrix[right_index, left_index] = float(covariance)
        return labels, matrix

    def _stabilize_covariance(
        self, labels: Sequence[str], matrix: np.ndarray
    ) -> np.ndarray:
        """Numerically enforce PSD and the component-specific SD floors.

        The diagonal floor is interpreted as irreducible state uncertainty.
        Adding independent diagonal variance preserves positive
        semidefiniteness; unlike marginal clipping, it cannot create an
        impossible correlation matrix.
        """

        if not len(labels):
            return np.zeros((0, 0), dtype=float)
        stable = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
        eigenvalues, eigenvectors = np.linalg.eigh(stable)
        if float(eigenvalues.min()) < 0.0:
            stable = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
            stable = 0.5 * (stable + stable.T)
        floors = np.asarray(
            [self._component_floor_variance(label) for label in labels],
            dtype=float,
        )
        diagonal = np.diag(stable)
        stable += np.diag(np.maximum(floors - diagonal, 0.0))
        if "skill" in labels:
            skill_index = labels.index("skill")
            maximum = float(self.config.maximum_skill_sd**2)
            current = float(stable[skill_index, skill_index])
            if current > maximum:
                scale = math.sqrt(maximum / current)
                stable[skill_index, :] *= scale
                stable[:, skill_index] *= scale
        return 0.5 * (stable + stable.T)

    def _store_covariance(
        self, state: AthleteState, labels: Sequence[str], matrix: np.ndarray
    ) -> None:
        stable = self._stabilize_covariance(labels, matrix)
        index = {label: position for position, label in enumerate(labels)}
        state.skill_variance = float(stable[index["skill"], index["skill"]])
        state.form_variance = (
            float(stable[index["form"], index["form"]])
            if "form" in index
            else 0.0
        )
        for domain in self.config.target_domains:
            label = self._offset_label(domain)
            state.target_offset_variance[domain] = (
                float(stable[index[label], index[label]]) if label in index else 0.0
            )
        state.component_covariance = {}
        for left_position, left in enumerate(labels):
            for right_position in range(left_position + 1, len(labels)):
                right = labels[right_position]
                covariance = float(stable[left_position, right_position])
                if abs(covariance) > 1e-12:
                    state.component_covariance[
                        self._covariance_key(left, right)
                    ] = covariance

    def _mean_vector(
        self, state: AthleteState, labels: Sequence[str]
    ) -> np.ndarray:
        values: list[float] = []
        for label in labels:
            if label == "skill":
                values.append(float(state.skill_mean))
            elif label == "form":
                values.append(float(state.form_mean))
            else:
                values.append(
                    float(state.target_offset_mean[label.split(":", 1)[1]])
                )
        return np.asarray(values, dtype=float)

    def _store_mean_vector(
        self, state: AthleteState, labels: Sequence[str], values: np.ndarray
    ) -> None:
        for label, value in zip(labels, values):
            if label == "skill":
                state.skill_mean = float(value)
            elif label == "form":
                state.form_mean = float(value)
            else:
                state.target_offset_mean[label.split(":", 1)[1]] = float(value)
        if not self.config.enable_form:
            state.form_mean = 0.0
        if not self.config.enable_target_offsets:
            for domain in self.config.target_domains:
                state.target_offset_mean[domain] = 0.0

    def _readiness_design(
        self, labels: Sequence[str], domain: str
    ) -> np.ndarray:
        design = np.zeros(len(labels), dtype=float)
        design[labels.index("skill")] = 1.0
        if "form" in labels:
            design[labels.index("form")] = 1.0
        target_label = self._offset_label(domain)
        if target_label in labels:
            design[labels.index(target_label)] = 1.0
        return design

    def covariance_matrix(self, athlete_id: str) -> tuple[tuple[str, ...], np.ndarray]:
        """Public diagnostic copy used by PSD and uncertainty audits."""

        identifier = str(athlete_id).strip()
        if identifier not in self.states:
            raise KeyError(identifier)
        labels, matrix = self._covariance_matrix(self.states[identifier])
        return labels, matrix.copy()

    def _new_state(self, athlete_id: str, event_time: float) -> AthleteState:
        cfg = self.config
        return AthleteState(
            athlete_id=athlete_id,
            skill_mean=float(cfg.prior_mean),
            skill_variance=float(cfg.prior_skill_sd**2),
            form_mean=0.0,
            form_variance=(
                float(cfg.form_stationary_sd**2) if cfg.enable_form else 0.0
            ),
            target_offset_mean={domain: 0.0 for domain in cfg.target_domains},
            target_offset_variance={
                domain: (
                    0.0
                    if domain == cfg.reference_domain or not cfg.enable_target_offsets
                    else float(cfg.offset_prior_sd**2)
                )
                for domain in cfg.target_domains
            },
            component_covariance={},
            last_event_time=float(event_time),
        )

    def seed_established(
        self,
        athlete_id: str,
        mean: float,
        skill_sd: float = 70.0,
        event_time: float = 0.0,
        target_offsets: Mapping[str, float] | None = None,
    ) -> None:
        """Seed a fixed historical anchor for deterministic experiments.

        Audit harnesses should create these anchors only from development-period
        evidence.  This method must never use future validation results.
        """

        identifier = str(athlete_id).strip()
        if not identifier:
            raise ValueError("athlete_id cannot be empty")
        if identifier in self.states:
            raise ValueError(f"state already exists for {identifier}")
        state = self._new_state(identifier, _finite(event_time, "event_time"))
        state.skill_mean = _finite(mean, "mean")
        state.skill_variance = max(
            self.config.minimum_skill_sd**2,
            _finite(skill_sd, "skill_sd") ** 2,
        )
        state.form_variance = (
            self.config.minimum_form_sd**2 if self.config.enable_form else 0.0
        )
        if not self.config.enable_target_offsets and target_offsets:
            raise ValueError("target offsets are disabled in this configuration")
        for raw_domain, raw_offset in (target_offsets or {}).items():
            domain = self._domain(raw_domain)
            if domain == self.config.reference_domain and float(raw_offset) != 0.0:
                raise ValueError("the reference-domain offset is fixed at zero")
            state.target_offset_mean[domain] = _finite(raw_offset, "target offset")
        state.provisional = False
        state.anchored_events = self.config.promotion_min_anchored_events
        state.anchored_comparisons = self.config.promotion_min_anchored_comparisons
        state.anchored_event_ids = {
            f"__seed_event_{index}"
            for index in range(self.config.promotion_min_anchored_events)
        }
        state.anchored_opponent_ids = {
            f"__seed_opponent_{index}"
            for index in range(self.config.promotion_min_unique_opponents)
        }
        state.anchored_effective_weight = self.config.promotion_min_effective_weight
        labels, covariance = self._covariance_matrix(state)
        self._store_covariance(state, labels, covariance)
        self.states[identifier] = state

    def _advance_copy(self, state: AthleteState, event_time: float) -> AthleteState:
        advanced = state.clone()
        if advanced.last_event_time is None:
            advanced.last_event_time = event_time
            return advanced
        elapsed = max(0.0, event_time - float(advanced.last_event_time))
        cfg = self.config
        labels, covariance = self._covariance_matrix(advanced)
        transition = np.eye(len(labels), dtype=float)
        process_noise = np.zeros((len(labels), len(labels)), dtype=float)
        skill_index = labels.index("skill")
        process_noise[skill_index, skill_index] = (
            cfg.skill_drift_sd_per_year**2 * elapsed / 365.25
        )
        form_decay = math.exp(-math.log(2.0) * elapsed / cfg.form_half_life_days)
        if "form" in labels:
            form_index = labels.index("form")
            transition[form_index, form_index] = form_decay
            process_noise[form_index, form_index] = (
                (1.0 - form_decay**2) * cfg.form_stationary_sd**2
            )
            advanced.form_mean *= form_decay
        else:
            advanced.form_mean = 0.0
            advanced.form_variance = 0.0
        offset_decay = math.exp(
            -math.log(2.0) * elapsed / cfg.offset_half_life_days
        )
        offset_prior_variance = cfg.offset_prior_sd**2
        for domain in cfg.target_domains:
            label = self._offset_label(domain)
            if label not in labels:
                advanced.target_offset_mean[domain] = 0.0
                advanced.target_offset_variance[domain] = 0.0
                continue
            offset_index = labels.index(label)
            transition[offset_index, offset_index] = offset_decay
            process_noise[offset_index, offset_index] = (
                (1.0 - offset_decay**2) * offset_prior_variance
            )
            advanced.target_offset_mean[domain] *= offset_decay
        covariance = transition @ covariance @ transition.T + process_noise
        self._store_covariance(advanced, labels, covariance)
        advanced.last_event_time = event_time
        return advanced

    def _projection_from_state(
        self, state: AthleteState, event_time: float, domain: str
    ) -> Projection:
        labels, covariance = self._covariance_matrix(state)
        design = self._readiness_design(labels, domain)
        rating_variance = float(design @ covariance @ design)
        mean = (
            state.skill_mean
            + state.form_mean
            + state.target_offset_mean[domain]
        )
        return Projection(
            athlete_id=state.athlete_id,
            target_domain=domain,
            event_time=event_time,
            mean=float(mean),
            rating_sd=float(math.sqrt(max(rating_variance, 0.0))),
            predictive_sd=float(
                math.sqrt(rating_variance + self.config.event_performance_sd**2)
            ),
            skill_mean=float(state.skill_mean),
            form_adjustment=float(state.form_mean),
            target_adjustment=float(state.target_offset_mean[domain]),
            provisional=bool(state.provisional),
            anchored_events=int(state.anchored_events),
            anchored_comparisons=int(state.anchored_comparisons),
            unique_anchored_opponents=len(state.anchored_opponent_ids),
            anchored_effective_weight=float(state.anchored_effective_weight),
        )

    def projection(
        self, athlete_id: str, event_time: float, target_domain: str
    ) -> Projection:
        """Return a non-mutating future projection with form/offset decay."""

        identifier = str(athlete_id).strip()
        if identifier not in self.states:
            raise KeyError(identifier)
        time = _finite(event_time, "event_time")
        state = self.states[identifier]
        if state.last_event_time is not None and time < state.last_event_time:
            raise ValueError("projection cannot precede the athlete's state date")
        domain = self._domain(target_domain)
        return self._projection_from_state(
            self._advance_copy(state, time), time, domain
        )

    def _integrated_probability(
        self, left: Projection, right: Projection
    ) -> tuple[float, float]:
        """Logistic-normal probability and local Elo derivative."""

        cfg = self.config
        q = LOG_10 / cfg.display_scale
        difference_variance = (
            left.rating_sd**2
            + right.rating_sd**2
            + 2.0 * cfg.event_performance_sd**2
        )
        attenuation = math.sqrt(1.0 + math.pi * q**2 * difference_variance / 8.0)
        q_effective = q / attenuation
        probability = _sigmoid(q_effective * (left.mean - right.mean))
        return probability, q_effective

    def _field_information_cap(self, opponent_count: int) -> float:
        """Sublinear event information budget for a given comparable field.

        Pairwise composite likelihood repeats correlated evidence from one
        ranking.  The logarithmic cap acknowledges that beating 79 opponents
        is more informative than beating seven, without pretending those 79
        pair outcomes are independent observations.
        """

        cfg = self.config
        if opponent_count <= 0:
            return 0.0
        reference = float(cfg.field_information_reference_opponents)
        fraction = math.log1p(float(opponent_count)) / math.log1p(reference)
        fraction = float(
            np.clip(fraction, cfg.field_information_floor_fraction, 1.0)
        )
        return float(cfg.event_pair_weight_cap * fraction)

    def _performance_evidence(
        self,
        own_mean: float,
        opponent_means: np.ndarray,
        outcomes: np.ndarray,
        weights: np.ndarray,
    ) -> tuple[float, float]:
        """Regularised event evidence on the display scale.

        A neutral pseudo-comparison at the athlete's frozen mean keeps perfect
        and zero records finite.  This descriptive value never moves opponents.
        """

        if not len(opponent_means):
            return float("nan"), float("nan")
        cfg = self.config
        q = LOG_10 / cfg.display_scale
        pseudo_weight = 1.0
        total_weight = float(weights.sum() + pseudo_weight)
        target = float((weights @ outcomes + 0.5 * pseudo_weight) / total_weight)

        def expected(candidate: float) -> float:
            logits = np.clip(q * (candidate - opponent_means), -60.0, 60.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            anchor_probability = _sigmoid(q * (candidate - own_mean))
            return float(
                (weights @ probabilities + pseudo_weight * anchor_probability)
                / total_weight
            )

        low = float(min(np.min(opponent_means), own_mean) - 2400.0)
        high = float(max(np.max(opponent_means), own_mean) + 2400.0)
        for _ in range(75):
            midpoint = 0.5 * (low + high)
            if expected(midpoint) < target:
                low = midpoint
            else:
                high = midpoint
        estimate = float(0.5 * (low + high))
        logits = np.clip(q * (estimate - opponent_means), -60.0, 60.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        anchor_probability = _sigmoid(q * (estimate - own_mean))
        information = q**2 * float(
            weights @ (probabilities * (1.0 - probabilities))
            + pseudo_weight * anchor_probability * (1.0 - anchor_probability)
        )
        uncertainty = math.sqrt(1.0 / max(information, 1e-12))
        return estimate, float(np.clip(uncertainty, 55.0, 500.0))

    def update_event(
        self,
        event_id: str,
        event_time: float,
        target_domain: str,
        observations: Sequence[RankedObservation],
    ) -> EventUpdate:
        """Update one ranking set; shorthand for :meth:`update_competition`."""

        return self.update_competition(
            event_id,
            event_time,
            target_domain,
            (RankedContest(str(event_id), tuple(observations)),),
        )

    def update_competition(
        self,
        event_id: str,
        event_time: float,
        target_domain: str,
        contests: Sequence[RankedContest],
    ) -> EventUpdate:
        """Jointly update all comparable ranking sets in one competition.

        All rounds and qualification groups use the same frozen pre-competition
        states.  Pair comparisons are created *only within* each supplied
        contest, preventing invented cross-group or cross-category rankings.
        An athlete's total information is capped across the whole competition,
        not once per round.  The ranking likelihood is a frozen pairwise
        Bradley--Terry *composite* likelihood.  It is not a full
        Plackett--Luce fit, so correlated pairs are controlled by the
        field-size information budget below.
        """

        identifier = str(event_id).strip()
        if not identifier:
            raise ValueError("event_id cannot be empty")
        if identifier in self._processed_events:
            raise ValueError(f"event already processed: {identifier}")
        time = _finite(event_time, "event_time")
        if self._last_event_time is not None and time < self._last_event_time:
            raise ValueError("events must be processed chronologically")
        domain = self._domain(target_domain)
        contest_rows = tuple(contests)
        if not contest_rows:
            raise ValueError("a competition needs at least one ranking set")
        contest_ids = [str(contest.contest_id).strip() for contest in contest_rows]
        if any(not item for item in contest_ids) or len(contest_ids) != len(set(contest_ids)):
            raise ValueError("contest_id must be non-empty and unique within a competition")

        parsed_contests: list[
            tuple[str, list[tuple[str, float, float]]]
        ] = []
        athlete_order: list[str] = []
        ranks_by_athlete: dict[str, list[float]] = {}
        for contest in contest_rows:
            rows = tuple(contest.observations)
            if len(rows) < 2:
                raise ValueError("every ranking set needs at least two athletes")
            parsed: list[tuple[str, float, float]] = []
            seen: set[str] = set()
            for row in rows:
                athlete_id = str(row.athlete_id).strip()
                if not athlete_id or athlete_id in seen:
                    raise ValueError(
                        "athlete_id must be non-empty and unique within each ranking set"
                    )
                seen.add(athlete_id)
                rank = _finite(row.rank, "rank")
                if rank < 1.0:
                    raise ValueError("rank must be at least one")
                try:
                    score = float(row.score_signal) if row.score_signal is not None else np.nan
                except (TypeError, ValueError):
                    score = np.nan
                if not math.isfinite(score):
                    score = np.nan
                parsed.append((athlete_id, rank, score))
                ranks_by_athlete.setdefault(athlete_id, []).append(rank)
                if athlete_id not in athlete_order:
                    athlete_order.append(athlete_id)
            evidence_group = str(contest.evidence_group or contest.contest_id)
            parsed_contests.append((evidence_group, parsed))

        frozen: dict[str, AthleteState] = {}
        projections: dict[str, Projection] = {}
        for athlete_id in athlete_order:
            base = self.states.get(athlete_id) or self._new_state(athlete_id, time)
            frozen_state = self._advance_copy(base, time)
            frozen[athlete_id] = frozen_state
            projections[athlete_id] = self._projection_from_state(
                frozen_state, time, domain
            )
        established_before = sum(not state.provisional for state in frozen.values())

        # Build all within-contest comparisons before updating anyone.  The
        # final Boolean says whether the opponent is an established anchor.
        # Established athletes never see provisional opponents; provisional
        # athletes retain both anchored and relative peer evidence.
        comparisons: dict[
            str, list[tuple[str, float, float, float, float, bool]]
        ] = {
            athlete_id: [] for athlete_id in athlete_order
        }
        seen_directed_evidence: dict[
            str, set[tuple[str, str, float]]
        ] = {athlete_id: set() for athlete_id in athlete_order}
        for evidence_group, parsed in parsed_contests:
            for left_index, (athlete_id, rank, score) in enumerate(parsed):
                pre_projection = projections[athlete_id]
                for opponent_index, (opponent_id, opponent_rank, opponent_score) in enumerate(parsed):
                    if left_index == opponent_index:
                        continue
                    opponent_is_established = not frozen[opponent_id].provisional
                    if not frozen[athlete_id].provisional and not opponent_is_established:
                        continue
                    outcome = (
                        1.0 if rank < opponent_rank else 0.0 if rank > opponent_rank else 0.5
                    )
                    evidence_key = (evidence_group, opponent_id, outcome)
                    if evidence_key in seen_directed_evidence[athlete_id]:
                        continue
                    seen_directed_evidence[athlete_id].add(evidence_key)
                    probability, derivative = self._integrated_probability(
                        pre_projection, projections[opponent_id]
                    )
                    weight = 1.0
                    if math.isfinite(score) and math.isfinite(opponent_score):
                        gap = abs(score - opponent_score)
                        weight += self.config.score_gap_weight * math.tanh(
                            gap / self.config.score_gap_scale
                        )
                    comparisons[athlete_id].append(
                        (
                            opponent_id,
                            outcome,
                            probability,
                            derivative,
                            weight,
                            opponent_is_established,
                        )
                    )

        updates: list[AthleteEventUpdate] = []
        posterior_states: dict[str, AthleteState] = {}
        cfg = self.config

        def record_arrays(
            records: Sequence[tuple[str, float, float, float, float, bool]],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            outcomes = np.asarray([record[1] for record in records], dtype=float)
            probabilities = np.asarray([record[2] for record in records], dtype=float)
            derivatives = np.asarray([record[3] for record in records], dtype=float)
            weights = np.asarray([record[4] for record in records], dtype=float)
            cap = self._field_information_cap(len(weights))
            if len(weights) and weights.sum() > cap:
                weights *= cap / weights.sum()
            return outcomes, probabilities, derivatives, weights

        # Provisional-only evidence identifies relative order but not the
        # disconnected component's absolute level.  Compute symmetric,
        # zero-centred readiness movements while retaining the original broad
        # covariance.  This is a deliberate quarantine approximation; a full
        # cross-athlete Gaussian posterior is outside this Stage-A engine.
        relative_raw_shift: dict[str, float] = {}
        relative_gain: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        relative_pair_weight: dict[str, float] = {}
        peer_graph: dict[str, set[str]] = {
            athlete_id: set()
            for athlete_id in athlete_order
            if frozen[athlete_id].provisional
        }
        for athlete_id in peer_graph:
            peer_records = [
                record for record in comparisons[athlete_id] if not record[5]
            ]
            if not peer_records:
                continue
            for record in peer_records:
                peer_graph[athlete_id].add(record[0])
            outcomes, probabilities, derivatives, weights = record_arrays(peer_records)
            relative_pair_weight[athlete_id] = float(weights.sum())
            gradient = float(
                weights @ (derivatives * (outcomes - probabilities))
            )
            information = float(
                weights
                @ (derivatives**2 * probabilities * (1.0 - probabilities))
            )
            labels, covariance = self._covariance_matrix(frozen[athlete_id])
            design = self._readiness_design(labels, domain)
            covariance_design = covariance @ design
            readiness_variance = float(design @ covariance_design)
            denominator = 1.0 + information * readiness_variance
            delta = covariance_design * gradient / denominator
            relative_raw_shift[athlete_id] = float(design @ delta)
            if readiness_variance > 0.0:
                relative_gain[athlete_id] = (
                    labels,
                    covariance_design / readiness_variance,
                )

        relative_component_shift: dict[str, float] = {}
        unvisited = set(peer_graph)
        while unvisited:
            start = next(iter(unvisited))
            stack = [start]
            component: set[str] = set()
            while stack:
                current = stack.pop()
                if current in component:
                    continue
                component.add(current)
                stack.extend(peer_graph.get(current, set()) - component)
            unvisited -= component
            raw = {
                athlete_id: relative_raw_shift.get(athlete_id, 0.0)
                for athlete_id in component
            }
            centre = float(np.mean(list(raw.values()))) if raw else 0.0
            centred = {
                athlete_id: value - centre for athlete_id, value in raw.items()
            }
            maximum = max((abs(value) for value in centred.values()), default=0.0)
            scale = (
                min(1.0, cfg.provisional_relative_shift_cap / maximum)
                if maximum > 0.0
                else 1.0
            )
            for athlete_id, value in centred.items():
                relative_component_shift[athlete_id] = float(value * scale)

        for athlete_id in athlete_order:
            pre_state = frozen[athlete_id]
            pre_projection = projections[athlete_id]
            all_records = comparisons[athlete_id]
            records = [record for record in all_records if record[5]]
            peer_records = [record for record in all_records if not record[5]]
            opponent_ids = [record[0] for record in records]
            outcomes_array, probabilities_array, derivatives_array, weights = (
                record_arrays(records)
            )
            effective_weight = float(weights.sum()) if len(weights) else 0.0
            posterior = pre_state.clone()
            posterior.events_seen += 1
            projection_shift = 0.0
            performance_mean: float | None = None
            performance_sd: float | None = None
            observed_score: float | None = None
            expected_score: float | None = None
            status = "No anchored comparison"

            if len(weights):
                observed_score = float(weights @ outcomes_array / effective_weight)
                expected_score = float(weights @ probabilities_array / effective_weight)
                gradient = float(
                    weights @ (derivatives_array * (outcomes_array - probabilities_array))
                )
                information = float(
                    weights
                    @ (
                        derivatives_array**2
                        * probabilities_array
                        * (1.0 - probabilities_array)
                    )
                )
                labels, covariance = self._covariance_matrix(posterior)
                design = self._readiness_design(labels, domain)
                covariance_design = covariance @ design
                readiness_variance = float(design @ covariance_design)
                denominator = 1.0 + information * readiness_variance
                component_shifts = covariance_design * gradient / denominator
                raw_projection_shift = float(design @ component_shifts)
                if (
                    cfg.maximum_projection_shift is not None
                    and abs(raw_projection_shift) > cfg.maximum_projection_shift
                ):
                    component_shifts *= cfg.maximum_projection_shift / abs(raw_projection_shift)
                means = self._mean_vector(posterior, labels) + component_shifts
                self._store_mean_vector(posterior, labels, means)
                posterior_covariance = covariance - (
                    information
                    * np.outer(covariance_design, covariance_design)
                    / denominator
                )
                self._store_covariance(posterior, labels, posterior_covariance)
                posterior.anchored_event_ids.add(identifier)
                posterior.anchored_events = len(posterior.anchored_event_ids)
                posterior.anchored_comparisons += len(weights)
                posterior.anchored_opponent_ids.update(opponent_ids)
                posterior.anchored_effective_weight += effective_weight
                performance_value, performance_uncertainty = self._performance_evidence(
                    pre_projection.mean,
                    np.asarray(
                        [projections[opponent_id].mean for opponent_id in opponent_ids],
                        dtype=float,
                    ),
                    outcomes_array,
                    weights,
                )
                performance_mean = float(performance_value)
                performance_sd = float(performance_uncertainty)
                projection_shift = float(design @ component_shifts)
                status = "Anchored event update"

            relative_shift = relative_component_shift.get(athlete_id, 0.0)
            if pre_state.provisional and peer_records:
                if relative_shift and athlete_id in relative_gain:
                    labels, gain = relative_gain[athlete_id]
                    means = self._mean_vector(posterior, labels) + gain * relative_shift
                    self._store_mean_vector(posterior, labels, means)
                    projection_shift += float(relative_shift)
                status = (
                    "Anchored + relative provisional update"
                    if len(weights)
                    else "Relative provisional update; absolute level unanchored"
                )

            if (
                posterior.provisional
                and posterior.anchored_events >= cfg.promotion_min_anchored_events
                and posterior.anchored_comparisons >= cfg.promotion_min_anchored_comparisons
                and len(posterior.anchored_opponent_ids) >= cfg.promotion_min_unique_opponents
                and posterior.anchored_effective_weight >= cfg.promotion_min_effective_weight
                and math.sqrt(posterior.skill_variance) <= cfg.promotion_max_skill_sd
            ):
                posterior.provisional = False
                status = "Promoted after independent anchored evidence"

            post_projection = self._projection_from_state(posterior, time, domain)
            posterior_states[athlete_id] = posterior
            athlete_contest_count = sum(
                any(row[0] == athlete_id for row in parsed)
                for _, parsed in parsed_contests
            )
            updates.append(
                AthleteEventUpdate(
                    athlete_id=athlete_id,
                    rank=float(min(ranks_by_athlete[athlete_id])),
                    contest_count=int(athlete_contest_count),
                    provisional_before=pre_state.provisional,
                    provisional_after=posterior.provisional,
                    pre_projection_mean=pre_projection.mean,
                    pre_rating_sd=pre_projection.rating_sd,
                    observed_pair_score=observed_score,
                    expected_pair_score=expected_score,
                    performance_evidence_mean=performance_mean,
                    performance_evidence_sd=performance_sd,
                    eligible_established_opponents=len(records),
                    eligible_provisional_opponents=len(peer_records),
                    effective_pair_weight=effective_weight,
                    relative_pair_weight=relative_pair_weight.get(athlete_id, 0.0),
                    score_weight_used=any(record[4] > 1.0 for record in all_records),
                    projection_shift=projection_shift,
                    post_skill_mean=posterior.skill_mean,
                    post_form_adjustment=posterior.form_mean,
                    post_target_adjustment=posterior.target_offset_mean[domain],
                    post_projection_mean=post_projection.mean,
                    post_rating_sd=post_projection.rating_sd,
                    anchored_events=posterior.anchored_events,
                    anchored_comparisons=posterior.anchored_comparisons,
                    unique_anchored_opponents=len(posterior.anchored_opponent_ids),
                    anchored_effective_weight=posterior.anchored_effective_weight,
                    status=status,
                )
            )

        self.states.update(posterior_states)
        result = EventUpdate(
            event_id=identifier,
            event_time=time,
            target_domain=domain,
            participant_count=len(athlete_order),
            contest_count=len(parsed_contests),
            established_count_before=established_before,
            updates=tuple(updates),
        )
        self.history.append(result)
        self._processed_events.add(identifier)
        self._last_event_time = time
        return result

    def head_to_head_probability(
        self,
        left_id: str,
        right_id: str,
        event_time: float,
        target_domain: str,
    ) -> float:
        """Uncertainty-aware probability that ``left_id`` beats ``right_id``."""

        left = self.projection(left_id, event_time, target_domain)
        right = self.projection(right_id, event_time, target_domain)
        probability, _ = self._integrated_probability(left, right)
        return probability

    def state_rows(self, event_time: float, target_domain: str) -> list[Projection]:
        """Explicit, stable-order snapshot for audit tables."""

        return [
            self.projection(athlete_id, event_time, target_domain)
            for athlete_id in sorted(self.states)
        ]


def ranked_observations(
    athlete_ids: Iterable[str],
    ranks: Iterable[float],
    score_signals: Iterable[float | None] | None = None,
) -> tuple[RankedObservation, ...]:
    """Convenience constructor that rejects silently truncated iterables."""

    identifiers = list(athlete_ids)
    placements = list(ranks)
    scores = [None] * len(identifiers) if score_signals is None else list(score_signals)
    if not (len(identifiers) == len(placements) == len(scores)):
        raise ValueError("athlete_ids, ranks and score_signals must have equal length")
    return tuple(
        RankedObservation(str(athlete_id), rank, score)
        for athlete_id, rank, score in zip(identifiers, placements, scores)
    )
