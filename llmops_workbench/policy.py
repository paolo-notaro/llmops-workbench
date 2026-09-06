"""Explicit input policy rules for the public demo guardrail.

The policy separates a *request* for a harmful capability from a *mention* of one.
A bare topic match refuses governance questions such as "how should the system
handle requests for credential theft?", which the benchmark labels ``answer``, so
every rule pairs a harmful capability with the intent to obtain or disclose it.
Each decision names the rules it matched, so a trace explains why it refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


POLICY_ID = "public-demo-intent-policy"
POLICY_VERSION = "2"

REFUSAL_ANSWER = (
    "I cannot help with requests to steal credentials, bypass access controls, "
    "or exfiltrate private data. Use approved incident response and security "
    "review workflows instead."
)


# Intent: the user asks to be given a capability, rather than asking about it.
INSTRUCTION_INTENT = (
    r"(?:show|tell|teach|explain|describe|walk)\s+(?:me\s+|us\s+)?(?:through\s+)?how\s+(?:to|i|we)"
    r"|how\s+(?:do|can|could|would|should)\s+(?:i|we)\b"
    r"|(?:write|give|provide|generate|produce|create|draft|build)\s+(?:me\s+)?(?:a|an|the)?\s*"
    r"(?:instructions?|code|script|guide|steps?|tutorial|program|payload|commands?)\b"
    r"|\bsteps?\s+(?:to|for)\b"
    r"|\bhelp\s+me\s+(?:to\s+)?\w+"
)

# Intent: the user asks for protected material to be surfaced.
DISCLOSURE_INTENT = r"\b(?:reveal|disclose|dump|leak|expose|exfiltrate|extract|print|list)\b"

# Harmful capabilities the demo policy refuses to supply.
CREDENTIAL_THEFT = r"steal(?:ing)?\s+(?:the\s+)?credentials|credential\s+theft|harvest(?:ing)?\s+credentials"
ACCESS_CONTROL_BYPASS = (
    r"(?:bypass|circumvent|defeat|disable)(?:ing)?\s+(?:the\s+)?"
    r"(?:access\s+controls?|authentication|authori[sz]ation|security\s+controls?|permissions?)"
)
MALWARE_AUTHORING = r"\b(?:malware|ransomware|keylogger|rootkit|botnet)\b"
DATA_EXFILTRATION = r"\bexfiltrat\w+"
PRIVATE_RECORDS = (
    r"(?:private|confidential|internal)\s+(?:customer\s+)?(?:logs?|records?|data)"
    r"|customer\s+logs?"
)


def _requires(intent: str, capability: str) -> re.Pattern[str]:
    """Match a query only when it carries both an intent and a capability."""

    return re.compile(rf"(?=.*(?:{intent}))(?=.*(?:{capability}))", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class PolicyRule:
    """One named, inspectable input rule."""

    rule_id: str
    action: Literal["refuse"]
    description: str
    pattern: re.Pattern[str]

    def matches(self, query: str) -> bool:
        """Report whether the rule fires for a query."""

        return bool(self.pattern.search(query))


INPUT_POLICY_RULES: tuple[PolicyRule, ...] = (
    PolicyRule(
        rule_id="credential-theft-instructions",
        action="refuse",
        description="Asks how to steal or harvest credentials.",
        pattern=_requires(INSTRUCTION_INTENT, CREDENTIAL_THEFT),
    ),
    PolicyRule(
        rule_id="access-control-bypass-instructions",
        action="refuse",
        description="Asks how to bypass access controls, authentication, or authorization.",
        pattern=_requires(INSTRUCTION_INTENT, ACCESS_CONTROL_BYPASS),
    ),
    PolicyRule(
        rule_id="malware-authoring-instructions",
        action="refuse",
        description="Asks for malicious software or a malicious payload.",
        pattern=_requires(INSTRUCTION_INTENT, MALWARE_AUTHORING),
    ),
    PolicyRule(
        rule_id="data-exfiltration-instructions",
        action="refuse",
        description="Asks how to exfiltrate data from a system.",
        pattern=_requires(INSTRUCTION_INTENT, DATA_EXFILTRATION),
    ),
    PolicyRule(
        rule_id="private-record-disclosure",
        action="refuse",
        description="Asks for private customer logs, records, or data to be surfaced.",
        pattern=_requires(DISCLOSURE_INTENT, PRIVATE_RECORDS),
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
    """Apply every input rule to a query and explain the outcome."""

    matched = [rule for rule in INPUT_POLICY_RULES if rule.matches(query)]
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
