"""Disable project billing when the project budget is exhausted."""

from __future__ import annotations

import base64
import binascii
import json
import math
import os
from typing import Any

import functions_framework
from google.cloud import billing_v1


@functions_framework.cloud_event
def stop_billing(cloud_event) -> None:
    """Unlink billing once reported cost reaches the configured budget."""

    cost, budget = _parse_budget_event(cloud_event)

    if cost < budget:
        return

    project_id = os.getenv("TARGET_PROJECT_ID", "").strip()
    if not project_id:
        raise RuntimeError("TARGET_PROJECT_ID must be configured before billing can be disabled")

    project_name = f"projects/{project_id}"
    client = billing_v1.CloudBillingClient()
    billing_info = client.get_project_billing_info(name=project_name)

    if billing_info.billing_account_name:
        client.update_project_billing_info(
            name=project_name,
            project_billing_info=billing_v1.ProjectBillingInfo(
                billing_account_name="",
            ),
        )


def _parse_budget_event(cloud_event: Any) -> tuple[float, float]:
    """Validate and decode the cost values in a budget notification."""

    event_data = getattr(cloud_event, "data", None)
    if not isinstance(event_data, dict):
        raise ValueError("CloudEvent data must be an object containing message.data")

    message = event_data.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("data"), str):
        raise ValueError("CloudEvent data must contain a base64-encoded message.data string")

    try:
        decoded = base64.b64decode(message["data"], validate=True).decode("utf-8")
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("CloudEvent message.data must contain valid base64-encoded JSON") from error

    if not isinstance(payload, dict):
        raise ValueError("Budget notification payload must be a JSON object")

    try:
        cost = float(payload["costAmount"])
        budget = float(payload["budgetAmount"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Budget notification must contain numeric costAmount and budgetAmount") from error

    if not math.isfinite(cost) or not math.isfinite(budget) or cost < 0 or budget <= 0:
        raise ValueError("Budget notification amounts must be finite, with cost non-negative and budget positive")

    return cost, budget
