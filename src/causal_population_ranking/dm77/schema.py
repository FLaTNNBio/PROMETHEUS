"""Data contract for the research operationalization of DM 77 need levels."""

from __future__ import annotations


DM77_LEVEL_LABELS = {
    1: "Persona in salute",
    2: "Complessita minima o limitata nel tempo",
    3: "Complessita media",
    4: "Complessita medio-alta con possibile fragilita sociale",
    5: "Complessita elevata con possibile fragilita sociale e non autosufficienza",
    6: "Cure palliative",
}

# All inputs must be measured before the index date. Treatment, outcomes, causal
# scores, and evaluation-only truth are intentionally absent from this contract.
DM77_INPUT_FIELDS = (
    "age",
    "condition_distinct",
    "prior_inpatient",
    "prior_emergency",
    "medication_distinct",
    "frailty_index",
    "functional_limitation_score",
    "social_fragility_score",
    "cognitive_impairment",
    "non_self_sufficiency",
    "caregiver_available",
    "housing_instability",
    "palliative_need",
)

DM77_FORBIDDEN_INPUT_PREFIXES = (
    "true_",
    "oracle_",
    "potential_outcome_",
)

DM77_FORBIDDEN_INPUT_COLUMNS = {
    "treatment",
    "treatment_level",
    "observed_outcome",
    "outcome",
    "causal_priority",
    "raw_score",
    "calibrated_benefit",
    "latent_rank",
}
