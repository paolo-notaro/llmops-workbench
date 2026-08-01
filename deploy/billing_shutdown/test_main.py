"""Unit tests for the emergency billing shutdown handler."""

from __future__ import annotations

import base64
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from main import stop_billing


def _event(cost: float, budget: float) -> SimpleNamespace:
    payload = base64.b64encode(
        json.dumps({"costAmount": cost, "budgetAmount": budget}).encode("utf-8")
    ).decode("ascii")
    return SimpleNamespace(data={"message": {"data": payload}})


@patch("main.billing_v1.CloudBillingClient")
def test_billing_stays_enabled_below_budget(client_type: MagicMock) -> None:
    stop_billing(_event(1.0, 2.0))
    client_type.assert_not_called()


@patch.dict("os.environ", {"TARGET_PROJECT_ID": "llmops-workbench"})
@patch("main.billing_v1.CloudBillingClient")
def test_billing_is_unlinked_at_budget(client_type: MagicMock) -> None:
    client = client_type.return_value
    client.get_project_billing_info.return_value.billing_account_name = "billingAccounts/test"

    stop_billing(_event(2.0, 2.0))

    client.update_project_billing_info.assert_called_once()


@pytest.mark.parametrize(
    ("event", "message"),
    [
        (SimpleNamespace(data={}), "message.data"),
        (SimpleNamespace(data={"message": {"data": "not base64"}}), "base64-encoded JSON"),
    ],
)
def test_invalid_events_raise_clear_errors(event: SimpleNamespace, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        stop_billing(event)


@patch.dict("os.environ", {}, clear=True)
@patch("main.billing_v1.CloudBillingClient")
def test_target_project_is_required_at_budget(client_type: MagicMock) -> None:
    with pytest.raises(RuntimeError, match="TARGET_PROJECT_ID must be configured"):
        stop_billing(_event(2.0, 2.0))

    client_type.assert_not_called()
