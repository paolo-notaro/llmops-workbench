"""Explicit input policy rules for the public demo guardrail.

The policy separates a *request* for a harmful capability from a *mention* of one.
Matching a bare topic refuses governance questions such as "how should the system
handle requests for credential theft?", which the benchmark labels ``answer``.
Searching the whole query for an intent and a capability independently is not
enough either: it treats "explain how to prevent credential theft" as a request,
because the two appear together without being related.

So each rule names one capability and is decided inside a single clause:

* an **action** capability ("steal credentials") refuses as a bare imperative, or
  when a nearby preceding phrase asks how to carry it out;
* an **object** capability ("private customer logs") refuses when a nearby
  preceding verb asks for it to be produced or handed over.

A capability introduced by defensive framing ("prevent", "detect", "handle") is
not a request for it. Every decision names the rules it matched, so a trace
explains why it refused.

Patterns deliberately avoid unbounded ``.*`` between the intent and the
capability: the guardrail runs on a public endpoint, and a query-wide
zero-width conjunction is retried at every offset, which is quadratic in the
length of the query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


POLICY_ID = "public-demo-intent-policy"
POLICY_VERSION = "3"

REFUSAL_ANSWER = (
    "I cannot help with requests to steal credentials, bypass access controls, "
    "or exfiltrate private data. Use approved incident response and security "
    "review workflows instead."
)

# A rule is decided within one clause, and only reads a bounded prefix before the
# capability, so cost stays linear in the length of the query.
CLAUSE_BOUNDARY = re.compile(r"[.;?!\n]+")
INTENT_WINDOW = 60

# The clause opens with the capability itself, allowing a short connective.
IMPERATIVE_PREFIX = re.compile(
    r"^(?:\s*(?:so|then|now|instead|also|and|but|please|first|next)\b[,\s]*)*$",
    re.IGNORECASE,
)

# Asking to be told how to carry an action out.
INSTRUCTION_INTENT = re.compile(
    r"(?:show|tell|teach|explain|describe|walk)\s+(?:me\s+|us\s+)?(?:through\s+)?how\s+(?:to|i|we)"
    r"|how\s+(?:do|can|could|would)\s+(?:i|we)\b"
    r"|(?:write|give|provide|generate|produce|create|draft|build)\s+(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:instructions?|code|script|guide|steps?|tutorial|program|payload|commands?)\b"
    r"|\bsteps?\s+(?:to|for)\b"
    r"|\bhelp\s+me\b",
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
    r"\b(?:prevent|protect|defend|detect|mitigate|avoid|block|stop|guard|harden|remediate"
    r"|respond|handle|audit|monitor|against)\b",
    re.IGNORECASE,
)

CREDENTIAL_THEFT = re.compile(
    r"steal(?:ing)?\s+(?:the\s+)?credentials|credential\s+theft|harvest(?:ing)?\s+credentials",
    re.IGNORECASE,
)
ACCESS_CONTROL_BYPASS = re.compile(
    r"(?:bypass|circumvent|defeat|disable)(?:ing)?\s+(?:the\s+)?"
    r"(?:access\s+controls?|authentication|authori[sz]ation|security\s+controls?|permissions?)",
    re.IGNORECASE,
)
DATA_EXFILTRATION = re.compile(r"\bexfiltrat\w+", re.IGNORECASE)
MALWARE = re.compile(r"\b(?:malware|ransomware|keylogger|rootkit|botnet)\b", re.IGNORECASE)
PRIVATE_RECORDS = re.compile(
    r"(?:private|confidential|internal)\s+(?:customer\s+)?(?:logs?|records?|data)"
    r"|customer\s+logs?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PolicyRule:
    """One named, inspectable input rule."""

    rule_id: str
    action: Literal["refuse"]
    description: str
    kind: Literal["action", "object"]
    capability: re.Pattern[str]

    def matches(self, clause: str) -> bool:
        """Report whether the clause requests this capability rather than mentioning it."""

        for found in self.capability.finditer(clause):
            prefix = clause[: found.start()]
            if DEFENSIVE_FRAMING.search(prefix):
                continue
            if self.kind == "action" and IMPERATIVE_PREFIX.match(prefix):
                return True
            window = prefix[-INTENT_WINDOW:]
            intent = INSTRUCTION_INTENT if self.kind == "action" else DELIVERY_INTENT
            if intent.search(window):
                return True
        return False


INPUT_POLICY_RULES: tuple[PolicyRule, ...] = (
    PolicyRule(
        rule_id="credential-theft-instructions",
        action="refuse",
        description="Asks to steal or harvest credentials.",
        kind="action",
        capability=CREDENTIAL_THEFT,
    ),
    PolicyRule(
        rule_id="access-control-bypass-instructions",
        action="refuse",
        description="Asks to bypass access controls, authentication, or authorization.",
        kind="action",
        capability=ACCESS_CONTROL_BYPASS,
    ),
    PolicyRule(
        rule_id="data-exfiltration-instructions",
        action="refuse",
        description="Asks to exfiltrate data from a system.",
        kind="action",
        capability=DATA_EXFILTRATION,
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
    """Apply every input rule clause by clause and explain the outcome."""

    clauses = [clause for clause in CLAUSE_BOUNDARY.split(query) if clause.strip()]
    matched = [
        rule for rule in INPUT_POLICY_RULES
        if any(rule.matches(clause) for clause in clauses)
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
