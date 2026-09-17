"""
MedVerify AI - Stage 11: Medical Safety Guardrail Service

Implements:
1. Emergency Symptom Detection: Identifies claims describing acute medical emergencies
   and routes to emergency guidance instead of verification.
2. Personal Medical Advice Classifier: NLP-based detection of inputs seeking
   personal diagnosis/treatment/dosage advice -> REFUSED_SAFETY.
3. Vulnerability Flagging: Detects pediatric, pregnancy, and elderly-specific claims
   for extra-caution messaging.
4. Context-Aware Medical Disclaimer Generator: Produces severity-appropriate disclaimers
   based on the verification verdict.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ENUMS & DATA CLASSES
# ---------------------------------------------------------------------------

class SafetyAction(str, Enum):
    """Action the guardrail recommends."""
    ALLOW = "ALLOW"
    REFUSE_PERSONAL = "REFUSE_PERSONAL"
    EMERGENCY_REDIRECT = "EMERGENCY_REDIRECT"
    ALLOW_WITH_CAUTION = "ALLOW_WITH_CAUTION"


class VulnerabilityFlag(str, Enum):
    """Vulnerable population flags."""
    PEDIATRIC = "PEDIATRIC"
    PREGNANCY = "PREGNANCY"
    ELDERLY = "ELDERLY"
    MENTAL_HEALTH = "MENTAL_HEALTH"
    NONE = "NONE"


class DisclaimerSeverity(str, Enum):
    """Disclaimer severity levels."""
    MILD = "MILD"
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    CRITICAL = "CRITICAL"


@dataclass
class SafetyResult:
    """Complete safety analysis result."""
    action: SafetyAction
    is_safe_to_verify: bool
    vulnerability_flags: List[VulnerabilityFlag] = field(default_factory=list)
    disclaimer: str = ""
    disclaimer_severity: DisclaimerSeverity = DisclaimerSeverity.MILD
    emergency_message: Optional[str] = None
    refusal_reason: Optional[str] = None
    matched_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EMERGENCY SYMPTOM PATTERNS
# ---------------------------------------------------------------------------

EMERGENCY_PATTERNS = [
    # Cardiac emergencies
    {
        "patterns": [
            r"\b(having|experiencing|feeling|have)\b.*\b(chest pain|heart attack|cardiac arrest|chest tightness|crushing chest)\b",
            r"\bchest\b.*\b(tight|pressure|crushing|squeezing|radiating)\b",
            r"\bheart\b.*\b(stopped|racing|pounding)\b.*\b(right now|currently|at this moment)\b",
            r"\bunremitting chest\b",
        ],
        "category": "cardiac_emergency",
        "message": "If you or someone else is experiencing chest pain or symptoms of a heart attack, call emergency services (911 / 112 / 999) immediately. Do not wait for online verification."
    },
    # Stroke symptoms
    {
        "patterns": [
            r"\b(having|experiencing|showing)\b.*\bstroke\b",
            r"\b(face drooping|arm weakness|speech difficulty|slurred speech)\b",
            r"\bcan'?t\b.*\b(move|feel|speak)\b.*\b(side|arm|leg|face)\b",
            r"\bvision suddenly went\b",
        ],
        "category": "stroke_emergency",
        "message": "If you suspect a stroke, use the FAST method (Face, Arms, Speech, Time) and call emergency services immediately. Every minute counts."
    },
    # Breathing / Pediatric Acute emergencies
    {
        "patterns": [
            r"\bcan'?t\b.*\bbreathe?\b",
            r"\b(severe|extreme|acute)\b.*\b(breathing|respiratory)\b.*\b(difficulty|distress|problem)\b",
            r"\b(choking|suffocating|gasping|blue lips)\b",
            r"\banaphyla(xis|ctic)\b.*\b(shock|reaction)\b",
            r"\b(temperature of 10[0-9]|having a seizure right now)\b",
            r"\bcoughing up blood\b",
        ],
        "category": "respiratory_emergency",
        "message": "If you are experiencing severe difficulty breathing or an acute pediatric crisis, call emergency services immediately."
    },
    # Self-harm / suicidal ideation
    {
        "patterns": [
            r"\b(want to|going to|planning to)\b.*\b(kill|harm|hurt|end)\b.*\b(myself|my life)\b",
            r"\bsuicid(e|al)\b",
            r"\bself[- ]harm\b",
        ],
        "category": "mental_health_crisis",
        "message": "If you or someone you know is in crisis, please reach out for help immediately: National Suicide Prevention Lifeline: 988 (US), Crisis Text Line: Text HOME to 741741, International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/"
    },
    # Poisoning / overdose
    {
        "patterns": [
            r"\b(overdos|poison|swallowed|ingested)\b.*\b(pills?|medication|chemical|bleach|detergent|paracetamol)\b",
            r"\btook (too (much|many)|an overdose)\b",
        ],
        "category": "poisoning_emergency",
        "message": "If you suspect poisoning or overdose, call Poison Control (1-800-222-1222 in the US) or emergency services immediately. Do not induce vomiting unless instructed by medical professionals."
    },
]

# ---------------------------------------------------------------------------
# PERSONAL MEDICAL ADVICE PATTERNS
# ---------------------------------------------------------------------------

PERSONAL_ADVICE_PATTERNS = [
    # Dosage / medication questions
    {
        "patterns": [
            r"\b(should|can|do)\s+i\b.*\b(take|stop|start|switch|increase|decrease|change|inject)\b.*\b(medication|medicine|drug|dose|dosage|prescription|pill|mg|units?|fast-acting)\b",
            r"\bhow (much|many|often)\b.*\b(should i|can i|to administer|take at one time)\b",
            r"\bwhat (dose|dosage|amount)\b.*\b(should i|of)\b",
            r"\bis it (safe|okay|ok)\b.*\bfor me to\b.*\b(take|stop|mix|combine)\b",
            r"\bcan i take\b.*\b(instead|mg)\b",
            r"\bhow many milligrams\b",
        ],
        "category": "dosage_advice",
        "reason": "This input seeks personal medication dosage or treatment advice. MedVerify AI verifies general medical claims but cannot provide individualized treatment recommendations. Please consult your healthcare provider for personalized advice."
    },
    # Diagnosis seeking
    {
        "patterns": [
            r"\bdo i have\b.*\b(disease|condition|disorder|syndrome|cancer|diabetes|infection)\b",
            r"\bwhat('s| is) wrong with me\b",
            r"\b(diagnose|diagnosis)\b.*\b(me|the condition)\b",
            r"\bi (think|feel|believe)\b.*\bi (have|might have|may have)\b",
            r"\bcould i have\b.*\b(disease|condition|disorder|syndrome)\b",
            r"\bdiagnose me\b",
            r"\btell me which illness i have\b",
            r"\bconfirm if i have it\b",
            r"\bwhat diagnosis does this suggest for me\b",
            r"\bdiagnose the condition\b",
        ],
        "category": "diagnosis_seeking",
        "reason": "This input seeks a personal medical diagnosis. MedVerify AI is designed to verify general medical claims against scientific evidence, not to diagnose individual conditions. Please consult a qualified healthcare professional."
    },
    # Treatment recommendations
    {
        "patterns": [
            r"\bwhat (medicine|drug|pharmaceutical drug|antibiotic|antibiotics|antidepressant|treatment|remedy)\b.*\b(should i|to take|is best)\b",
            r"\bwhat should i\b.*\b(do|take|try|use|prescribe)\b.*\bfor (my|a|stage)\b",
            r"\bshould i (go|see|visit|discontinue)\b.*\b(doctor|hospital|ER|emergency|chemotherapy|medication)\b",
            r"\bshould i be (worried|concerned)\b.*\babout my\b",
            r"\bwhat (treatment|therapy|remedy|home remedies)\b.*\b(for my|should i|give me)\b",
            r"\bmy (doctor|physician)\b.*\bshould i\b.*\b(follow|listen|trust|ignore)\b",
            r"\b(give me a personalized prescription|can you prescribe|prescribe me)\b",
            r"\btell me (what|which) (medicine|drug|antibiotic|antibiotics|antidepressant) i should take\b",
            r"\bwhich antidepressant\b.*\bshould i\b",
        ],
        "category": "treatment_advice",

        "reason": "This input seeks personalized treatment recommendations. MedVerify AI verifies general medical claims against peer-reviewed evidence and cannot replace professional medical consultation. Please discuss treatment options with your healthcare provider."
    },
]

# ---------------------------------------------------------------------------
# VULNERABILITY DETECTION PATTERNS
# ---------------------------------------------------------------------------

VULNERABILITY_PATTERNS = {
    VulnerabilityFlag.PEDIATRIC: [
        r"\b(child|children|infant|baby|babies|toddler|pediatric|paediatric|newborn|neonatal)\b",
        r"\b(kid|kids)\b.*\b(health|medicine|drug|vaccine|treatment)\b",
        r"\bunder\s+(the\s+)?age\s+of\s+(1[0-8]|[0-9])\b",
        r"\b(adolescent|teenager|teen|puberty)\b",
    ],
    VulnerabilityFlag.PREGNANCY: [
        r"\b(pregnan|prenatal|antenatal|postnatal|postpartum|breastfeed|nursing|lactating)\b",
        r"\b(fetus|fetal|foetal|unborn|trimester)\b",
        r"\b(expecting|expectant)\s+(mother|mom|woman)\b",
        r"\b(maternal|maternity)\b.*\b(health|care|risk)\b",
    ],
    VulnerabilityFlag.ELDERLY: [
        r"\b(elderly|geriatric|senior|aged|aging|ageing)\b",
        r"\bover\s+(the\s+)?age\s+of\s+(6[5-9]|[7-9][0-9]|100)\b",
        r"\b(old age|older (adult|patient|people|person))\b",
        r"\b(dementia|alzheimer)\b",
    ],
    VulnerabilityFlag.MENTAL_HEALTH: [
        r"\b(depress(ion|ed|ive)|anxiety|bipolar|schizophren|ptsd|ocd|panic\s+disorder)\b",
        r"\b(mental\s+(health|illness|disorder)|psychiatric)\b",
        r"\b(antidepress|anxiolytic|psychotropic|ssri|snri|benzodiazepine)\b",
    ],
}


# ---------------------------------------------------------------------------
# DISCLAIMER TEMPLATES
# ---------------------------------------------------------------------------

DISCLAIMER_TEMPLATES = {
    DisclaimerSeverity.MILD: (
        "Medical Information Notice: This verification is based on currently available "
        "scientific evidence and is provided for informational purposes only. While the evidence "
        "supports this claim, medical knowledge evolves continuously. Always consult your healthcare "
        "provider before making health decisions."
    ),
    DisclaimerSeverity.MODERATE: (
        "Important Medical Disclaimer: The scientific evidence for this claim is limited "
        "or mixed. This means experts have not reached a consensus, or there is insufficient research "
        "to draw a definitive conclusion. Do not make health decisions based solely on this "
        "verification. Consult a qualified healthcare professional."
    ),
    DisclaimerSeverity.STRONG: (
        "Critical Medical Warning: This claim contradicts current peer-reviewed medical "
        "evidence. Following health advice based on contradicted claims may pose serious risks "
        "to your health. Please consult your doctor or a qualified healthcare professional before "
        "acting on any health information."
    ),
    DisclaimerSeverity.CRITICAL: (
        "Urgent Medical Safety Notice: This claim involves a vulnerable population or "
        "high-risk medical scenario. Extra caution is warranted. The information provided here "
        "is for general awareness only. Seek immediate professional medical advice, especially "
        "when children, pregnant individuals, elderly patients, or mental health concerns are involved."
    ),
}

VULNERABILITY_CAUTION_ADDENDUM = {
    VulnerabilityFlag.PEDIATRIC: (
        " Pediatric Caution: This claim involves children or infants. Pediatric medicine "
        "differs significantly from adult medicine — dosages, risks, and treatment guidelines vary by "
        "age and weight. Always consult a pediatrician."
    ),
    VulnerabilityFlag.PREGNANCY: (
        " Pregnancy Caution: This claim involves pregnancy, prenatal, or breastfeeding "
        "considerations. Many medications and interventions carry different risk profiles during "
        "pregnancy. Consult your OB-GYN or midwife before making any health decisions."
    ),
    VulnerabilityFlag.ELDERLY: (
        " Geriatric Caution: This claim involves elderly or aging populations. Older adults "
        "may have different pharmacokinetics, comorbidities, and risk factors. Consult a geriatric "
        "medicine specialist."
    ),
    VulnerabilityFlag.MENTAL_HEALTH: (
        " Mental Health Caution: This claim involves mental health conditions or psychotropic "
        "medications. Mental health treatment is highly individualized. Never change psychiatric "
        "medication without consulting your prescribing psychiatrist."
    ),
}


# ---------------------------------------------------------------------------
# SAFETY GUARDRAIL ENGINE
# ---------------------------------------------------------------------------

class MedicalSafetyGuardrail:
    """
    Stage 11: Comprehensive medical safety guardrail engine.

    Runs three detection passes on every incoming claim:
    1. Emergency symptom detection (highest priority)
    2. Personal medical advice detection
    3. Vulnerability population flagging
    """

    def __init__(self):
        # Pre-compile all regex patterns for performance
        self._emergency_compiled = []
        for entry in EMERGENCY_PATTERNS:
            compiled = [re.compile(p, re.IGNORECASE) for p in entry["patterns"]]
            self._emergency_compiled.append({
                "compiled": compiled,
                "category": entry["category"],
                "message": entry["message"],
            })

        self._personal_compiled = []
        for entry in PERSONAL_ADVICE_PATTERNS:
            compiled = [re.compile(p, re.IGNORECASE) for p in entry["patterns"]]
            self._personal_compiled.append({
                "compiled": compiled,
                "category": entry["category"],
                "reason": entry["reason"],
            })

        self._vulnerability_compiled = {}
        for flag, patterns in VULNERABILITY_PATTERNS.items():
            self._vulnerability_compiled[flag] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

        logger.info("[SafetyGuardrail] Initialized with %d emergency, %d personal advice, %d vulnerability patterns.",
                     len(self._emergency_compiled), len(self._personal_compiled), len(self._vulnerability_compiled))

    # ------------------------------------------------------------------
    # PASS 1: Emergency Detection
    # ------------------------------------------------------------------

    def _detect_emergency(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Check if the input describes an active medical emergency.
        Returns (category, emergency_message) or None.
        """
        for entry in self._emergency_compiled:
            for pattern in entry["compiled"]:
                if pattern.search(text):
                    logger.warning("[SafetyGuardrail] EMERGENCY detected: %s", entry["category"])
                    return entry["category"], entry["message"]
        return None

    # ------------------------------------------------------------------
    # PASS 2: Personal Advice Detection
    # ------------------------------------------------------------------

    def _detect_personal_advice(self, text: str) -> Optional[Tuple[str, str]]:
        """
        Check if the input seeks personal medical advice.
        Returns (category, refusal_reason) or None.
        """
        for entry in self._personal_compiled:
            for pattern in entry["compiled"]:
                if pattern.search(text):
                    logger.info("[SafetyGuardrail] Personal advice detected: %s", entry["category"])
                    return entry["category"], entry["reason"]
        return None

    # ------------------------------------------------------------------
    # PASS 3: Vulnerability Flagging
    # ------------------------------------------------------------------

    def _detect_vulnerabilities(self, text: str) -> List[VulnerabilityFlag]:
        """
        Detect if the claim involves vulnerable populations.
        Returns list of matched vulnerability flags.
        """
        flags = []
        for flag, patterns in self._vulnerability_compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    flags.append(flag)
                    break  # One match per flag is enough
        return flags

    # ------------------------------------------------------------------
    # PUBLIC API: Full Safety Analysis
    # ------------------------------------------------------------------

    def analyze(self, raw_text: str) -> SafetyResult:
        """
        Run the complete 3-pass safety analysis on a raw claim text.

        Priority order:
        1. Emergency detection -> EMERGENCY_REDIRECT
        2. Personal advice detection -> REFUSE_PERSONAL
        3. Vulnerability flagging -> ALLOW_WITH_CAUTION (or ALLOW)
        """
        matched_patterns = []

        # Pass 1: Emergency check (highest priority)
        emergency_result = self._detect_emergency(raw_text)
        if emergency_result:
            category, message = emergency_result
            matched_patterns.append(f"emergency:{category}")
            return SafetyResult(
                action=SafetyAction.EMERGENCY_REDIRECT,
                is_safe_to_verify=False,
                emergency_message=message,
                disclaimer=DISCLAIMER_TEMPLATES[DisclaimerSeverity.CRITICAL],
                disclaimer_severity=DisclaimerSeverity.CRITICAL,
                matched_patterns=matched_patterns,
            )

        # Pass 2: Personal advice check
        personal_result = self._detect_personal_advice(raw_text)
        if personal_result:
            category, reason = personal_result
            matched_patterns.append(f"personal_advice:{category}")
            return SafetyResult(
                action=SafetyAction.REFUSE_PERSONAL,
                is_safe_to_verify=False,
                refusal_reason=reason,
                disclaimer=DISCLAIMER_TEMPLATES[DisclaimerSeverity.STRONG],
                disclaimer_severity=DisclaimerSeverity.STRONG,
                matched_patterns=matched_patterns,
            )

        # Pass 3: Vulnerability flagging
        vuln_flags = self._detect_vulnerabilities(raw_text)
        if vuln_flags:
            for f in vuln_flags:
                matched_patterns.append(f"vulnerability:{f.value}")

            action = SafetyAction.ALLOW_WITH_CAUTION
            disclaimer = DISCLAIMER_TEMPLATES[DisclaimerSeverity.CRITICAL]
            for f in vuln_flags:
                addendum = VULNERABILITY_CAUTION_ADDENDUM.get(f, "")
                if addendum:
                    disclaimer += addendum

            return SafetyResult(
                action=action,
                is_safe_to_verify=True,
                vulnerability_flags=vuln_flags,
                disclaimer=disclaimer,
                disclaimer_severity=DisclaimerSeverity.CRITICAL,
                matched_patterns=matched_patterns,
            )

        # No safety concerns detected
        return SafetyResult(
            action=SafetyAction.ALLOW,
            is_safe_to_verify=True,
            disclaimer=DISCLAIMER_TEMPLATES[DisclaimerSeverity.MILD],
            disclaimer_severity=DisclaimerSeverity.MILD,
        )

    def get_verdict_disclaimer(
        self,
        verdict: str,
        vulnerability_flags: Optional[List[VulnerabilityFlag]] = None,
    ) -> Tuple[str, DisclaimerSeverity]:
        """
        Generate a context-aware disclaimer based on the verification verdict
        and any vulnerability flags.
        """
        # Determine base severity from verdict
        verdict_lower = verdict.lower().strip()
        if verdict_lower in ("contradicted", "refuted"):
            severity = DisclaimerSeverity.STRONG
        elif verdict_lower in ("insufficient evidence", "mixed", "inconclusive"):
            severity = DisclaimerSeverity.MODERATE
        elif verdict_lower in ("supported",):
            severity = DisclaimerSeverity.MILD
        else:
            severity = DisclaimerSeverity.MODERATE

        # Escalate to CRITICAL if vulnerable populations involved
        if vulnerability_flags:
            severity = DisclaimerSeverity.CRITICAL

        disclaimer = DISCLAIMER_TEMPLATES[severity]

        # Append vulnerability-specific addenda
        if vulnerability_flags:
            for flag in vulnerability_flags:
                addendum = VULNERABILITY_CAUTION_ADDENDUM.get(flag, "")
                if addendum:
                    disclaimer += addendum

        return disclaimer, severity


# ---------------------------------------------------------------------------
# MODULE-LEVEL SINGLETON
# ---------------------------------------------------------------------------

_guardrail_instance: Optional[MedicalSafetyGuardrail] = None


def get_safety_guardrail() -> MedicalSafetyGuardrail:
    """Get or create the singleton safety guardrail instance."""
    global _guardrail_instance
    if _guardrail_instance is None:
        _guardrail_instance = MedicalSafetyGuardrail()
    return _guardrail_instance


def evaluate_clinical_safety(text: str) -> dict:
    """
    Convenience evaluator for safety test suites & API routing.
    Returns dict:
      - is_permitted: bool
      - action: str
      - refusal_reasons: list of str
    """
    guard = get_safety_guardrail()
    res = guard.analyze(text)
    is_permitted = res.is_safe_to_verify

    reasons = []
    if res.refusal_reason:
        reasons.append(res.refusal_reason)
    if res.emergency_message:
        reasons.append(res.emergency_message)

    return {
        "is_permitted": is_permitted,
        "action": res.action.value,
        "refusal_reasons": reasons,
        "disclaimer_severity": res.disclaimer_severity.value,
        "vulnerability_flags": [f.value for f in res.vulnerability_flags],
    }

