"""
scoring.py — Deterministic readiness score calculator for PharmaAI Dx.

Score is derived purely from dimension_scores and initiative_description
in AgentState. No LLM is involved — identical inputs always produce
identical outputs.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING LOGIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Severity → Points:
    Low          → 100
    Medium       →  50
    High         →   0
    Not assessed →   0  (penalises incomplete diagnostics)

  Final score = Σ (severity_points[dim] × weight[dim])  → rounded int
  Range: 0–100.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTEXT-AWARE WEIGHTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Initiative type is detected deterministically via keyword matching on
  initiative_description. Five profiles are supported:

  clinical_decision_support — AI surfacing recommendations to clinicians
    at point of care. Regulatory and change management risk dominate.

  patient_facing — AI interacting directly with patients.
    Regulatory and governance risk are critical.

  drug_discovery — AI for target ID, molecule screening, biomarkers.
    Data quality and technical architecture dominate.

  regulatory_ops — AI assisting submissions, dossiers, CSR review.
    Regulatory alignment and governance are primary.

  internal_ops — AI for supply chain, forecasting, back-office.
    Technical production-readiness and scale design dominate.

  If no profile matches, balanced default weights are applied.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REPRODUCIBILITY GUARANTEE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Detection → pure string matching (lowercased).
  Scoring   → deterministic weighted average.
  No randomness. No LLM. No external calls.
  Same inputs → same type → same weights → same score.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Severity → points mapping
# ---------------------------------------------------------------------------

SEVERITY_POINTS: dict[str, int] = {
    "Low":          100,
    "Medium":        50,
    "High":           0,
    "Not assessed":   0,
}

# ---------------------------------------------------------------------------
# Score band thresholds  (min_score_inclusive, label, hex_colour)
# ---------------------------------------------------------------------------

BAND_THRESHOLDS: list[tuple[int, str, str]] = [
    (80, "High Readiness",     "#66bb6a"),
    (55, "Moderate Readiness", "#ffa726"),
    (30, "Low Readiness",      "#ef5350"),
    (0,  "Critical Risk",      "#b71c1c"),
]

# ---------------------------------------------------------------------------
# Initiative type profiles
# ---------------------------------------------------------------------------

INITIATIVE_PROFILES: list[dict] = [
    {
        "type": "clinical_decision_support",
        "label": "Clinical Decision Support",
        "description": (
            "Regulatory Alignment and Change Management are upweighted: "
            "SaMD misclassification and clinician non-adoption are the "
            "dominant failure modes at point-of-care."
        ),
        "keywords": [
            "prescrib", "oncologist", "clinician", "point of care",
            "clinical decision", "clinical workflow", "adverse drug",
            "adr", "diagnostic support", "treatment recommendation",
            "physician", "clinical alert", "ehr alert", "emr alert",
            "radiolog", "patholog", "triage ai",
        ],
        "weights": {
            "Regulatory Alignment":       0.25,
            "Change Management":          0.22,
            "Data Readiness":             0.20,
            "Governance & Ownership":     0.15,
            "Technical Architecture Fit": 0.12,
            "Pilot-to-Scale Design":      0.06,
        },
    },
    {
        "type": "patient_facing",
        "label": "Patient-Facing Application",
        "description": (
            "Regulatory Alignment and Governance are upweighted: "
            "direct patient interaction creates the highest regulatory "
            "and accountability stakes."
        ),
        "keywords": [
            "patient-facing", "patient facing", "patient app",
            "consumer app", "patient chatbot", "patient portal",
            "wearable", "remote monitoring", "direct to patient",
            "patient engagement", "mobile health", "mhealth",
        ],
        "weights": {
            "Regulatory Alignment":       0.28,
            "Governance & Ownership":     0.22,
            "Change Management":          0.18,
            "Data Readiness":             0.16,
            "Technical Architecture Fit": 0.10,
            "Pilot-to-Scale Design":      0.06,
        },
    },
    {
        "type": "drug_discovery",
        "label": "Drug Discovery / Research AI",
        "description": (
            "Data Readiness and Technical Architecture are upweighted: "
            "model validity depends entirely on training data quality "
            "and architectural fit to complex biological problems."
        ),
        "keywords": [
            "drug discovery", "molecule", "target identification",
            "compound screen", "biomarker discovery", "genomic",
            "protein folding", "de novo design", "hit identification",
            "lead optimisation", "lead optimization", "qsar",
            "in silico", "omics", "transcriptomic", "proteomic",
            "phenotypic screen",
        ],
        "weights": {
            "Data Readiness":             0.28,
            "Technical Architecture Fit": 0.25,
            "Governance & Ownership":     0.18,
            "Pilot-to-Scale Design":      0.13,
            "Regulatory Alignment":       0.10,
            "Change Management":          0.06,
        },
    },
    {
        "type": "regulatory_ops",
        "label": "Regulatory Operations",
        "description": (
            "Regulatory Alignment and Governance are upweighted: "
            "submission tools must be fully auditable and validated "
            "against ICH/FDA/EMA standards."
        ),
        "keywords": [
            "regulatory submission", "dossier", "clinical study report",
            "csr", "ich ", "cmc", "regulatory affairs", "ectd",
            "label review", "labelling ai", "pharmacovigilance",
            "signal detection", "adverse event report", "psur",
            "rmp", "safety report", "regulatory document",
        ],
        "weights": {
            "Regulatory Alignment":       0.28,
            "Governance & Ownership":     0.22,
            "Data Readiness":             0.20,
            "Change Management":          0.15,
            "Technical Architecture Fit": 0.10,
            "Pilot-to-Scale Design":      0.05,
        },
    },
    {
        "type": "internal_ops",
        "label": "Internal Operations / Productivity",
        "description": (
            "Technical Architecture and Pilot-to-Scale are upweighted: "
            "lower regulatory stakes shift risk toward production "
            "engineering and scale readiness."
        ),
        "keywords": [
            "supply chain", "demand forecast", "inventory",
            "back-office", "back office", "internal tool",
            "internal productivity", "manufacturing optimis",
            "manufacturing optimiz", "quality control", "batch release",
            "procurement", "finance ai", "hr ai", "workforce",
            "document review internal", "internal workflow",
        ],
        "weights": {
            "Technical Architecture Fit": 0.25,
            "Pilot-to-Scale Design":      0.22,
            "Data Readiness":             0.20,
            "Governance & Ownership":     0.15,
            "Change Management":          0.12,
            "Regulatory Alignment":       0.06,
        },
    },
]

DEFAULT_PROFILE: dict = {
    "type": "general",
    "label": "General Pharma AI",
    "description": (
        "Balanced weights applied — initiative type could not be "
        "determined from the description provided."
    ),
    "keywords": [],
    "weights": {
        "Data Readiness":             0.20,
        "Regulatory Alignment":       0.20,
        "Governance & Ownership":     0.18,
        "Technical Architecture Fit": 0.17,
        "Change Management":          0.15,
        "Pilot-to-Scale Design":      0.10,
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_initiative_type(initiative_description: str) -> dict:
    """
    Detect initiative profile via keyword matching (first match wins).
    Always returns a profile dict — never None.

    Negation guard: if a keyword appears immediately after 'not', 'non-',
    'no ', or 'without ' it is treated as a non-match to avoid false
    positives like "not patient-facing" triggering patient_facing.
    """
    import re
    text = initiative_description.lower()

    def _keyword_present(kw: str, txt: str) -> bool:
        """Return True only if kw appears and is not in a negated context."""
        if kw not in txt:
            return False
        # Check each occurrence — if all are negated, treat as absent
        for m in re.finditer(re.escape(kw), txt):
            start = m.start()
            # Look at up to 10 chars before the match for negation words
            prefix = txt[max(0, start - 10):start]
            if re.search(r'\b(not|non|no |without )\s*$', prefix):
                continue  # this occurrence is negated
            return True   # found at least one non-negated occurrence
        return False

    for profile in INITIATIVE_PROFILES:
        if any(_keyword_present(kw, text) for kw in profile["keywords"]):
            return profile
    return DEFAULT_PROFILE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_readiness_score(
    dimension_scores: dict[str, str],
    initiative_description: str = "",
) -> int:
    """
    Calculate and return the overall readiness score (0–100).

    Args:
        dimension_scores:       Dimension name → severity string.
        initiative_description: Free-text used to detect initiative type
                                and select weights. Pass "" for defaults.

    Returns:
        Integer in [0, 100]. Fully deterministic.
    """
    profile = _detect_initiative_type(initiative_description)
    weights = profile["weights"]
    total = 0.0
    for dim, weight in weights.items():
        severity = dimension_scores.get(dim, "Not assessed")
        points = SEVERITY_POINTS.get(severity, 0)
        total += points * weight
    return round(total)


def get_score_band(score: int) -> tuple[str, str]:
    """Return (label, hex_colour) for a given score."""
    for threshold, label, colour in BAND_THRESHOLDS:
        if score >= threshold:
            return label, colour
    return BAND_THRESHOLDS[-1][1], BAND_THRESHOLDS[-1][2]


def get_initiative_profile(initiative_description: str) -> dict:
    """
    Return the full detected profile dict for a description.
    Useful for surfacing the type label and rationale in the UI.
    """
    return _detect_initiative_type(initiative_description)


def score_breakdown(
    dimension_scores: dict[str, str],
    initiative_description: str = "",
) -> list[dict]:
    """
    Per-dimension breakdown ordered by descending weighted contribution.
    Each row: dimension, severity, points, weight, weighted_contribution.
    """
    profile = _detect_initiative_type(initiative_description)
    weights = profile["weights"]
    rows = []
    for dim, weight in weights.items():
        severity = dimension_scores.get(dim, "Not assessed")
        points = SEVERITY_POINTS.get(severity, 0)
        rows.append({
            "dimension":             dim,
            "severity":              severity,
            "points":                points,
            "weight":                weight,
            "weighted_contribution": round(points * weight, 2),
        })
    return sorted(rows, key=lambda r: r["weighted_contribution"], reverse=True)
