"""
Deterministic, evidence-driven retention planning.

Shared by both data modes. The synthetic model scores 27 features and the real
model scores 57, and the two name the same underlying idea differently --
`days_since_last_listen` versus `log_days_since_last_listen`,
`prior_cancellations` versus `cancel_count`. Rules are therefore written
against CONCEPTS, each resolved to whichever concrete column the current
subscriber actually has. One rule set, two feature spaces, no duplication and
no mode-specific branching in the callers.

TWO HARD RULES
--------------
1. No recommendation ever fires from probability alone. Probability decides
   *intensity* (via config.recommendation_severity); observed feature evidence
   decides *what to do*. A LOW-risk subscriber with one ugly feature receives
   maintenance guidance, never a retention intervention -- that asymmetry was a
   real defect in the previous engine and is now enforced by the severity gate.
2. Every piece of evidence is a value actually read off this subscriber's
   feature row. Nothing is inferred, rounded into a nicer story, or carried
   over from another subscriber.

Contributions quoted alongside evidence are the model's own log-odds
attributions. They describe how the model reached its estimate. They are not
causal claims and the wording never implies otherwise.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Iterable

import pandas as pd

from src import config

log = logging.getLogger("recommendations")

# ------------------------------------------------------------------ concepts
#
# Each concept lists candidate columns in preference order. The first one
# present on the row with a usable value wins.
CONCEPTS: dict[str, list[str]] = {
    "active_days_30d": ["active_days_last_30d", "log_active_days_last_30d"],
    "active_days_90d": ["active_days_last_90d", "log_active_days_last_90d"],
    "activity_rate_30d": ["log_activity_rate_last_30d"],
    "secs_30d": ["total_secs_last_30d", "log_secs_last_30d"],
    "avg_daily_secs": ["avg_daily_secs", "log_average_daily_secs"],
    "activity_trend": ["engagement_trend", "log_active_days_trend_delta"],
    "secs_trend_ratio": ["log_secs_trend_ratio"],
    "activity_ratio": ["recent_activity_ratio"],
    "days_since_listen": ["days_since_last_listen", "log_days_since_last_listen"],
    "unique_songs": ["unique_songs_last_30d", "log_average_unique_songs_per_active_day"],
    "completion": ["completion_ratio", "log_completion_rate"],
    "cancellations": ["prior_cancellations", "cancel_count"],
    "cancel_rate": ["cancel_rate"],
    "recent_cancels": ["cancels_last_30d"],
    "auto_renew": ["is_auto_renew", "current_auto_renew"],
    "auto_renew_rate": ["auto_renew_rate"],
    "txn_recency": ["recency_days", "days_since_last_transaction"],
    "txn_freq": ["transaction_frequency"],
    "txns_30d": ["txns_last_30d"],
    "discount": ["mean_discount"],
    "tenure": ["tenure_days"],
    "expiry": ["days_until_membership_expiry"],
}

# Human labels for evidence lines. Registered by each data mode at start-up so
# the engine does not need to know which feature space it is looking at.
LABELS: dict[str, str] = {}


def register_labels(labels: dict[str, str]) -> None:
    LABELS.update(labels)


# How a concept's value is phrased in evidence lines.
UNITS: dict[str, str] = {
    "secs_30d": "seconds", "avg_daily_secs": "seconds/day",
    "days_since_listen": "days", "txn_recency": "days", "tenure": "days",
    "expiry": "days", "activity_rate_30d": "ratio", "completion": "ratio",
    "cancel_rate": "ratio", "auto_renew_rate": "ratio",
    "secs_trend_ratio": "x", "activity_ratio": "x",
}


class Resolver:
    """Reads concept values off one subscriber's feature row."""

    def __init__(self, row: pd.Series, signals: list[dict] | None = None,
                 formatter: Callable[[str, Any], str] | None = None):
        self.row = row
        self._contrib = {s["feature"]: float(s.get("contribution", 0.0))
                         for s in (signals or [])}
        # Each data mode renders its own columns: the real service knows that
        # current_auto_renew is a flag and log_secs_last_30d is seconds. Without
        # this the evidence line reads "current_auto_renew = 0.00", which is
        # accurate and useless.
        self._format = formatter

    def column(self, concept: str) -> str | None:
        for candidate in CONCEPTS.get(concept, []):
            if candidate in self.row.index:
                value = self.row.get(candidate)
                if value is not None and not (isinstance(value, float) and pd.isna(value)):
                    return candidate
        return None

    def value(self, concept: str) -> float | None:
        col = self.column(concept)
        if col is None:
            return None
        try:
            return float(self.row[col])
        except (TypeError, ValueError):
            return None

    def contribution(self, concept: str) -> float:
        col = self.column(concept)
        return self._contrib.get(col, 0.0) if col else 0.0

    def evidence(self, concept: str, fmt: str = "{:.2f}") -> dict[str, Any] | None:
        """One evidence line: the real column name and the real value."""
        col = self.column(concept)
        value = self.value(concept)
        if col is None or value is None:
            return None
        shown = (self._format(col, self.row[col]) if self._format
                 else fmt.format(value))
        return {
            "feature": col,
            "label": LABELS.get(col, col),
            "value": shown,
            "raw": value,
            "unit": UNITS.get(concept, ""),
            "contribution": round(self.contribution(concept), 4),
        }


# ------------------------------------------------------------------ rules

class Rule:
    """One evidence-backed recommendation.

    `when` inspects resolved concept values and returns True only if this
    subscriber genuinely exhibits the behaviour. `build` then assembles the
    rationale from those same values, so the text can never disagree with the
    evidence beside it.
    """

    def __init__(self, key: str, title: str, intent: str, concepts: list[str],
                 when: Callable[[Resolver], bool],
                 rationale: Callable[[Resolver], str],
                 next_step: str, weight: float,
                 min_severity: str = "EARLY_ENGAGEMENT"):
        self.key = key
        self.title = title
        self.intent = intent
        self.concepts = concepts
        self.when = when
        self.rationale = rationale
        self.next_step = next_step
        self.weight = weight
        self.min_severity = min_severity

    def fires(self, r: Resolver) -> bool:
        try:
            return bool(self.when(r))
        except Exception as e:                                    # pragma: no cover
            log.warning("rule %s failed: %s", self.key, e)
            return False

    def build(self, r: Resolver) -> dict[str, Any]:
        evidence = [e for e in (r.evidence(c) for c in self.concepts) if e]
        strength = max((abs(e["contribution"]) for e in evidence), default=0.0)
        return {
            "key": self.key,
            "title": self.title,
            "intent": self.intent,
            "rationale": self.rationale(r),
            "evidence": evidence,
            "next_step": self.next_step,
            "signal_strength": round(strength, 4),
            "weight": self.weight,
        }


def _pct(value: float) -> str:
    return f"{value * 100:.0f}%"


RULES: list[Rule] = [
    Rule(
        key="reengage_listening",
        title="Re-engage listening activity",
        intent="Recover declining engagement before the renewal decision",
        concepts=["active_days_30d", "activity_trend", "days_since_listen"],
        when=lambda r: (r.value("active_days_30d") is not None
                        and r.value("active_days_30d") < 8),
        rationale=lambda r: (
            f"Listening activity is low: {r.value('active_days_30d'):.0f} active "
            f"day(s) in the 30 days before the observation cutoff"
            + (f", and {r.value('days_since_listen'):.0f} day(s) have passed since "
               f"the last recorded session"
               if r.value("days_since_listen") is not None else "")
            + "."),
        next_step=("Consider a personalised content prompt built from this "
                   "subscriber's own recent listening history."),
        weight=4.0,
    ),
    Rule(
        key="declining_trend",
        title="Address the downward engagement trend",
        intent="Intervene while the decline is still shallow",
        concepts=["activity_trend", "secs_trend_ratio", "secs_30d"],
        when=lambda r: ((r.value("activity_trend") is not None
                         and r.value("activity_trend") < 0)
                        or (r.value("secs_trend_ratio") is not None
                            and r.value("secs_trend_ratio") < 0.8)),
        rationale=lambda r: (
            (f"Listening time in the last 30 days is "
             f"{_pct(r.value('secs_trend_ratio'))} of the previous 30 days."
             if r.value("secs_trend_ratio") is not None else
             f"Engagement is trending downward "
             f"({r.value('activity_trend'):.1f} over the trailing window).")),
        next_step=("Consider a light-touch re-engagement message before the "
                   "trend deepens."),
        weight=3.6,
    ),
    Rule(
        key="dormant",
        title="Reach a dormant subscriber",
        intent="Re-establish contact after an extended silence",
        concepts=["days_since_listen", "active_days_30d"],
        when=lambda r: (r.value("days_since_listen") is not None
                        and r.value("days_since_listen") >= 14),
        rationale=lambda r: (
            f"No listening activity recorded for "
            f"{r.value('days_since_listen'):.0f} days before the cutoff."),
        next_step=("Consider a win-back contact; routine in-app prompts are "
                   "unlikely to reach a dormant subscriber."),
        weight=4.2,
    ),
    Rule(
        key="shallow_listening",
        title="Improve session depth",
        intent="Convert short sessions into sustained listening",
        concepts=["completion", "unique_songs", "avg_daily_secs"],
        when=lambda r: (r.value("completion") is not None
                        and r.value("completion") < 0.55),
        rationale=lambda r: (
            f"Only {_pct(r.value('completion'))} of plays run to completion, "
            f"which suggests the current recommendations are not landing."),
        next_step=("Consider refreshing discovery surfaces or playlist "
                   "recommendations for this subscriber."),
        weight=2.6,
    ),
    Rule(
        key="narrow_catalogue",
        title="Broaden catalogue exposure",
        intent="Reduce reliance on a narrow set of tracks",
        concepts=["unique_songs", "active_days_30d"],
        when=lambda r: (r.value("unique_songs") is not None
                        and r.value("unique_songs") < 12),
        rationale=lambda r: (
            f"Unique-track variety is low at {r.value('unique_songs'):.1f}, "
            f"which often precedes disengagement."),
        next_step="Consider introducing adjacent genres or curated discovery.",
        weight=2.2,
    ),
    Rule(
        key="auto_renew_off",
        title="Send a renewal reminder",
        intent="Prevent a lapse that requires no decision to occur",
        concepts=["auto_renew", "expiry", "txn_recency"],
        when=lambda r: (r.value("auto_renew") is not None
                        and r.value("auto_renew") == 0),
        rationale=lambda r: (
            "Auto-renewal is disabled, so the membership lapses unless it is "
            "renewed manually"
            + (f"; the current term ends in {r.value('expiry'):.0f} day(s)."
               if r.value("expiry") is not None and r.value("expiry") >= 0
               else ".")),
        next_step=("Consider a renewal reminder timed ahead of the expiry "
                   "date, with the option to re-enable auto-renewal."),
        weight=5.0,
    ),
    Rule(
        key="cancellation_history",
        title="Route to a retention specialist",
        intent="Match a repeat-cancel profile with a human conversation",
        concepts=["cancellations", "cancel_rate", "recent_cancels"],
        when=lambda r: (r.value("cancellations") is not None
                        and r.value("cancellations") > 0),
        rationale=lambda r: (
            f"{r.value('cancellations'):.0f} prior cancellation(s) on record"
            + (f", a {_pct(r.value('cancel_rate'))} cancellation rate across all "
               f"transactions." if r.value("cancel_rate") is not None else ".")),
        next_step=("Consider routing to a retention specialist rather than an "
                   "automated flow."),
        weight=4.4,
    ),
    Rule(
        key="payment_lapse",
        title="Review billing continuity",
        intent="Catch a lapse caused by payment friction rather than intent",
        concepts=["txn_recency", "txn_freq", "txns_30d"],
        when=lambda r: (r.value("txn_recency") is not None
                        and r.value("txn_recency") >= 45),
        rationale=lambda r: (
            f"No transaction recorded for {r.value('txn_recency'):.0f} days, "
            f"which may indicate a billing problem rather than a decision to "
            f"leave."),
        next_step=("Consider verifying the payment method before treating this "
                   "as a retention case."),
        weight=3.2,
    ),
    Rule(
        key="expiring_soon",
        title="Act before the term ends",
        intent="Use the remaining term as the intervention window",
        concepts=["expiry", "auto_renew"],
        when=lambda r: (r.value("expiry") is not None
                        and 0 <= r.value("expiry") <= 14),
        rationale=lambda r: (
            f"The current membership term ends in "
            f"{r.value('expiry'):.0f} day(s), which bounds how long any "
            f"intervention has to work."),
        next_step="Prioritise contact inside the remaining term.",
        weight=3.8,
    ),
]

# LOW-risk subscribers get maintenance observations instead of interventions.
MAINTENANCE: list[Rule] = [
    Rule(
        key="maintain_auto_renew",
        title="Auto-renewal is active",
        intent="Preserve the strongest stabilising factor",
        concepts=["auto_renew", "auto_renew_rate"],
        when=lambda r: (r.value("auto_renew") is not None
                        and r.value("auto_renew") == 1),
        rationale=lambda r: (
            "Auto-renewal is enabled, which the model associates with lower "
            "predicted churn for this subscriber."),
        next_step="No action indicated. Avoid flows that risk disabling it.",
        weight=3.0, min_severity="NONE",
    ),
    Rule(
        key="maintain_activity",
        title="Listening activity is healthy",
        intent="Keep the current engagement level",
        concepts=["active_days_30d", "secs_30d", "activity_rate_30d"],
        when=lambda r: (r.value("active_days_30d") is not None
                        and r.value("active_days_30d") >= 15),
        rationale=lambda r: (
            f"{r.value('active_days_30d'):.0f} active day(s) in the trailing "
            f"30 days, well above the level where the model begins to flag "
            f"disengagement."),
        next_step="Continue normal engagement; monitor for a sustained decline.",
        weight=2.8, min_severity="NONE",
    ),
    Rule(
        key="maintain_tenure",
        title="Established subscription history",
        intent="Recognise a long-standing relationship",
        concepts=["tenure", "txn_freq"],
        when=lambda r: r.value("tenure") is not None and r.value("tenure") >= 365,
        rationale=lambda r: (
            f"{r.value('tenure') / 365.0:.1f} year(s) of membership history, a "
            f"profile the model associates with stability."),
        next_step="No action indicated.",
        weight=2.0, min_severity="NONE",
    ),
    Rule(
        key="maintain_clean_record",
        title="No cancellation history",
        intent="Note the absence of a known risk marker",
        concepts=["cancellations"],
        when=lambda r: (r.value("cancellations") is not None
                        and r.value("cancellations") == 0),
        rationale=lambda r: (
            "No prior cancellations on record, one of the markers the model "
            "weighs most heavily."),
        next_step="No action indicated.",
        weight=1.8, min_severity="NONE",
    ),
]

SEVERITY_ORDER = ["NONE", "EARLY_ENGAGEMENT", "TARGETED", "HIGH_PRIORITY", "CRITICAL"]

SEVERITY_LABELS = {
    "NONE": "Stable",
    "EARLY_ENGAGEMENT": "Early engagement",
    "TARGETED": "Targeted",
    "HIGH_PRIORITY": "High priority",
    "CRITICAL": "Critical priority",
}

PLAN_HEADINGS = {
    "NONE": "Subscriber appears stable",
    "EARLY_ENGAGEMENT": "Suggested engagement steps",
    "TARGETED": "Recommended retention actions",
    "HIGH_PRIORITY": "Priority retention actions",
    "CRITICAL": "Priority retention actions",
}


def _flagged_reasons(r: Resolver, signals: list[dict] | None,
                     limit: int = 3) -> list[dict[str, Any]]:
    """Why the model flagged this subscriber: its own strongest risk-raising
    contributions, with the observed value beside each."""
    raising = [s for s in (signals or []) if float(s.get("contribution", 0)) > 0]
    raising.sort(key=lambda s: -float(s["contribution"]))
    return [{
        "feature": s["feature"],
        "label": s.get("label", s["feature"]),
        "value": s.get("value"),
        "contribution": round(float(s["contribution"]), 4),
    } for s in raising[:limit]]


def build_plan(probability: float, row: pd.Series,
               signals: list[dict] | None = None,
               max_actions: int = 4,
               formatter: Callable[[str, Any], str] | None = None) -> dict[str, Any]:
    """The personalised retention plan for one subscriber.

    Returns the policy envelope, why the model flagged them, and the ranked
    evidence-backed actions. Two subscribers with different feature profiles
    get different plans; two with identical profiles get identical plans.
    """
    severity = config.recommendation_severity(probability)
    band = config.risk_level(probability)
    intervening = severity in config.INTERVENTION_SEVERITIES
    resolver = Resolver(row, signals, formatter)

    pool: Iterable[Rule] = RULES if intervening else MAINTENANCE
    fired = [rule for rule in pool if rule.fires(resolver)]

    built = [rule.build(resolver) for rule in fired]
    # Rank by the model's own attribution first, then rule weight. A rule the
    # model actually leans on outranks one it does not, even if the latter
    # looks more dramatic.
    built.sort(key=lambda a: (-a["signal_strength"], -a["weight"]))
    actions = built[:max_actions]

    for position, action in enumerate(actions, start=1):
        action["priority"] = position
        action["severity"] = severity
        action["severity_label"] = SEVERITY_LABELS[severity]
        action.pop("weight", None)

    return {
        "risk_level": band,
        "severity": severity,
        "severity_label": SEVERITY_LABELS[severity],
        "heading": PLAN_HEADINGS[severity],
        "intervention": intervening,
        "probability": round(float(probability), 6),
        "flagged_because": _flagged_reasons(resolver, signals) if intervening else [],
        "actions": actions,
        "action_count": len(actions),
        "no_rule_matched": len(actions) == 0,
        "disclaimer": (
            "Actions are deterministic rules over this subscriber's observed "
            "feature values. Contributions describe how the model reached its "
            "estimate, not a causal mechanism, and nothing here claims an "
            "action will prevent churn."),
    }
