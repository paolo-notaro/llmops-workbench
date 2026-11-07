"""Explicit input policy rules for the public demo guardrail.

The policy separates a *request* for a harmful capability from a *mention* of one.
Matching a bare topic refuses governance questions such as "how should the system
handle requests for credential theft?", which the benchmark labels ``answer``.
Searching the whole query for an intent and a capability independently is not
enough either: it treats "explain how to prevent credential theft" as a request,
because the two appear together without being related.

So each rule names one capability and relates it to request intent:

* an **action** capability ("steal credentials") refuses as a genuine bare
  imperative, or when a preceding phrase asks how to carry it out;
* an **object** capability ("private customer logs") refuses when a preceding
  verb in the same clause asks for it to be produced or handed over.

A capability directly governed by defensive framing ("prevent credential
theft", "detect malware") is not a request for it. Defensive vocabulary
elsewhere in the request is not trusted as a blanket bypass. Every decision
names the rules it matched, so a trace explains why it refused.

The matcher normalizes whitespace before scanning, so line breaks cannot split
an intent phrase. It advances through precomputed intent, defensive, and clause
events once per rule; repeated capability terms therefore remain linear rather
than rebuilding and rescanning every prefix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


POLICY_ID = "public-demo-intent-policy"
POLICY_VERSION = "4"

REFUSAL_ANSWER = (
    "I cannot help with requests to steal credentials, bypass access controls, "
    "or exfiltrate private data. Use approved incident response and security "
    "review workflows instead."
)

# Newlines are formatting, not a semantic trust boundary. They are normalized
# to spaces before these patterns run.
CLAUSE_BOUNDARY = re.compile(r"[.;?!]+")
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?", re.IGNORECASE)

# The clause opens with the capability itself, allowing a short connective.
IMPERATIVE_PREFIX = re.compile(
    r"^[\W_]*(?:(?:so|then|now|instead|also|and|but|please|first|next)\b[\W_]*)*$",
    re.IGNORECASE,
)

# Asking to be told how to carry an action out.
INSTRUCTION_INTENT = re.compile(
    r"(?:show|tell|teach|explain|describe|walk)[\W_]+(?:me[\W_]+|us[\W_]+)?"
    r"(?:through[\W_]+)?how[\W_]+(?:to|i|we)\b"
    r"|how[\W_]+(?:do|can|could|would)[\W_]+(?:i|we)\b"
    r"|(?:write|give|provide|generate|produce|create|draft|build)[\W_]+"
    r"(?:me[\W_]+)?(?:(?:a|an|the)[\W_]+)?"
    r"(?:instructions?|code|script|guide|steps?|tutorial|program|payload|commands?)\b"
    r"|\bsteps?[\W_]+(?:to|for)\b"
    r"|\bhelp[\W_]+me\b",
    re.IGNORECASE,
)

# Asking for protected material to be produced or handed over.
DELIVERY_INTENT = re.compile(
    r"\b(?:show|give|send|provide|share|hand|email|upload|export|fetch|retrieve|pull|list|print"
    r"|dump|reveal|disclose|leak|expose|exfiltrate|extract"
    r"|write|create|build|generate|produce|develop)\b",
    re.IGNORECASE,
)

# Framing that discusses a capability in order to defend against it.
DEFENSIVE_FRAMING = re.compile(
    r"\b(?:prevent(?:s|ed|ing)?|protect(?:s|ed|ing)?|defend(?:s|ed|ing)?"
    r"|detect(?:s|ed|ing)?|mitigat(?:e|es|ed|ing)|avoid(?:s|ed|ing)?"
    r"|block(?:s|ed|ing)?|stop(?:s|ped|ping)?|guard(?:s|ed|ing)?"
    r"|harden(?:s|ed|ing)?|remediat(?:e|es|ed|ing)|respond(?:s|ed|ing)?"
    r"|handl(?:e|es|ed|ing)|audit(?:s|ed|ing)?|monitor(?:s|ed|ing)?|against)\b",
    re.IGNORECASE,
)

CREDENTIAL_THEFT = re.compile(
    r"steal(?:ing)?\s+(?:the\s+)?credentials|credential\s+theft|harvest(?:ing)?\s+credentials",
    re.IGNORECASE,
)
ACCESS_CONTROL_BYPASS = re.compile(
    r"(?:bypass(?:ing)?|circumvent(?:ing)?|defeat(?:ing)?|disabl(?:e|ing))\s+(?:the\s+)?"
    r"(?:access\s+controls?|authentication|authori[sz]ation|security\s+controls?|permissions?)",
    re.IGNORECASE,
)
DATA_EXFILTRATION = re.compile(r"\bexfiltrat(?:e|es|ed|ing|ion)\b", re.IGNORECASE)
MALWARE = re.compile(r"\b(?:malware|ransomware|keylogger|rootkit|botnet)\b", re.IGNORECASE)
PRIVATE_RECORDS = re.compile(
    r"(?:private|confidential|internal)\s+(?:customer\s+)?(?:logs?|records?|data)"
    r"|customer\s+logs?",
    re.IGNORECASE,
)

# Only short function-word bridges may carry a request across punctuation. This
# catches formatting such as "show me. the private logs" without connecting an
# unrelated earlier sentence to a later capability. Same-clause intent has no
# character window, so verbose but direct requests remain covered.
MAX_CROSS_CLAUSE_BRIDGE = 80
CROSS_CLAUSE_WORDS = frozenset(
    {"a", "an", "any", "for", "me", "please", "the", "to", "us"}
)
DIRECT_DEFENSE_WORDS = frozenset(
    {"a", "an", "against", "any", "from", "our", "the", "their", "this", "that", "your"}
)


@dataclass(frozen=True)
class _PolicyContext:
    """Precomputed query events shared by all rules."""

    text: str
    clause_boundaries: tuple[re.Match[str], ...]
    instruction_intents: tuple[re.Match[str], ...]
    delivery_intents: tuple[re.Match[str], ...]
    defensive_frames: tuple[re.Match[str], ...]

    @classmethod
    def build(cls, query: str) -> _PolicyContext:
        text = re.sub(r"\s+", " ", query).strip()
        return cls(
            text=text,
            clause_boundaries=tuple(CLAUSE_BOUNDARY.finditer(text)),
            instruction_intents=tuple(INSTRUCTION_INTENT.finditer(text)),
            delivery_intents=tuple(DELIVERY_INTENT.finditer(text)),
            defensive_frames=tuple(DEFENSIVE_FRAMING.finditer(text)),
        )


def _words(text: str) -> tuple[str, ...]:
    """Return normalized words from a small, already bounded span."""

    return tuple(match.group().lower() for match in WORD.finditer(text))


def _is_cross_clause_bridge(text: str, start: int, end: int) -> bool:
    """Allow punctuation to format a request, but not join unrelated clauses."""

    if end - start > MAX_CROSS_CLAUSE_BRIDGE:
        return False
    return set(_words(text[start:end])).issubset(CROSS_CLAUSE_WORDS)


def _is_directly_defensive(text: str, defense_end: int, capability_start: int) -> bool:
    """Report whether a defensive verb directly governs the capability."""

    if capability_start - defense_end > MAX_CROSS_CLAUSE_BRIDGE:
        return False
    return set(_words(text[defense_end:capability_start])).issubset(DIRECT_DEFENSE_WORDS)


def _is_bare_imperative(text: str, clause_start: int, capability_start: int) -> bool:
    """Recognize a short polite/connective prefix without rescanning a clause."""

    if capability_start - clause_start > MAX_CROSS_CLAUSE_BRIDGE:
        return False
    return IMPERATIVE_PREFIX.fullmatch(text[clause_start:capability_start]) is not None


@dataclass(frozen=True)
class PolicyRule:
    """One named, inspectable input rule."""

    rule_id: str
    action: Literal["refuse"]
    description: str
    kind: Literal["action", "object"]
    capability: re.Pattern[str]
    imperative_verbs: tuple[str, ...] = ()

    def matches(self, context: _PolicyContext) -> bool:
        """Report whether the query requests this capability rather than mentioning it."""

        intents = (
            context.instruction_intents
            if self.kind == "action"
            else context.delivery_intents
        )
        intent_index = 0
        defense_index = 0
        boundary_index = 0
        latest_intent: re.Match[str] | None = None
        latest_defense: re.Match[str] | None = None
        clause_start = 0

        for found in self.capability.finditer(context.text):
            while intent_index < len(intents) and intents[intent_index].end() <= found.start():
                latest_intent = intents[intent_index]
                intent_index += 1
            while (
                defense_index < len(context.defensive_frames)
                and context.defensive_frames[defense_index].end() <= found.start()
            ):
                latest_defense = context.defensive_frames[defense_index]
                defense_index += 1
            while (
                boundary_index < len(context.clause_boundaries)
                and context.clause_boundaries[boundary_index].end() <= found.start()
            ):
                clause_start = context.clause_boundaries[boundary_index].end()
                boundary_index += 1

            if latest_defense and _is_directly_defensive(
                context.text, latest_defense.end(), found.start()
            ):
                continue

            first_word = next(iter(_words(found.group())), "")
            if (
                self.kind == "action"
                and first_word in self.imperative_verbs
                and _is_bare_imperative(context.text, clause_start, found.start())
            ):
                return True

            if latest_intent and (
                latest_intent.end() >= clause_start
                or _is_cross_clause_bridge(context.text, latest_intent.end(), found.start())
            ):
                return True
        return False


INPUT_POLICY_RULES: tuple[PolicyRule, ...] = (
    PolicyRule(
        rule_id="credential-theft-instructions",
        action="refuse",
        description="Asks to steal or harvest credentials.",
        kind="action",
        capability=CREDENTIAL_THEFT,
        imperative_verbs=("harvest", "steal"),
    ),
    PolicyRule(
        rule_id="access-control-bypass-instructions",
        action="refuse",
        description="Asks to bypass access controls, authentication, or authorization.",
        kind="action",
        capability=ACCESS_CONTROL_BYPASS,
        imperative_verbs=("bypass", "circumvent", "defeat", "disable"),
    ),
    PolicyRule(
        rule_id="data-exfiltration-instructions",
        action="refuse",
        description="Asks to exfiltrate data from a system.",
        kind="action",
        capability=DATA_EXFILTRATION,
        imperative_verbs=("exfiltrate",),
    ),
    PolicyRule(
        rule_id="malware-authoring-instructions",
        action="refuse",
        description="Asks for malicious software or a malicious payload.",
        kind="object",
        capability=MALWARE,
    ),
    PolicyRule(
        rule_id="private-record-disclosure",
        action="refuse",
        description="Asks for private customer logs, records, or data to be surfaced.",
        kind="object",
        capability=PRIVATE_RECORDS,
    ),
)


@dataclass(frozen=True)
class PolicyDecision:
    """The input policy outcome for one query."""

    action: Literal["refuse", "answer"]
    matched_rules: list[str]
    reason: str

    @property
    def refuses(self) -> bool:
        """Report whether generation must be skipped."""

        return self.action == "refuse"


def decide_input_policy(query: str) -> PolicyDecision:
    """Apply every input rule to one precomputed query context."""

    context = _PolicyContext.build(query)
    matched = [
        rule for rule in INPUT_POLICY_RULES
        if rule.matches(context)
    ]
    if not matched:
        return PolicyDecision(
            action="answer",
            matched_rules=[],
            reason="No input policy rule matched.",
        )
    return PolicyDecision(
        action="refuse",
        matched_rules=[rule.rule_id for rule in matched],
        reason="; ".join(rule.description for rule in matched),
    )

