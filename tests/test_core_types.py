from __future__ import annotations

import pytest

from agent_runtime.core.types import (
    ModelConfigV0,
    RunEventPayloadV0,
    build_model_config_v0,
    parse_model_config,
    parse_run_event_payload,
)


def test_build_model_config_v0_defaults() -> None:
    config = build_model_config_v0(model="gpt-4o")
    assert isinstance(config, ModelConfigV0)
    assert config.version == "v0"
    assert config.provider == "openai"
    assert config.model == "gpt-4o"
    assert config.config["temperature"] == 0.0


def test_build_model_config_v0_includes_optional_fields() -> None:
    config = build_model_config_v0(
        model="gpt-4o-mini",
        temperature=0.3,
        api_key="secret",
        extra_config={"timeout": 30, "top_p": 0.95},
    )
    assert config.config["temperature"] == 0.3
    assert config.config["api_key"] == "secret"
    assert config.config["timeout"] == 30
    assert config.config["top_p"] == 0.95


def test_parse_model_config_from_mapping() -> None:
    parsed = parse_model_config(
        {
            "version": "v0",
            "provider": "openai",
            "model": "gpt-4o",
            "config": {"temperature": 0.2},
        }
    )
    assert isinstance(parsed, ModelConfigV0)
    assert parsed.config["temperature"] == 0.2


def test_parse_model_config_unknown_version_raises() -> None:
    with pytest.raises(ValueError):
        parse_model_config(
            {
                "version": "v99",
                "provider": "openai",
                "model": "gpt-4o",
                "config": {},
            }
        )


def test_parse_run_event_payload_defaults_to_v0() -> None:
    payload = parse_run_event_payload({"data": {"run_id": "r1"}})
    assert isinstance(payload, RunEventPayloadV0)
    assert payload.version == "v0"
    assert payload.data["run_id"] == "r1"


def test_parse_run_event_payload_unknown_version_raises() -> None:
    with pytest.raises(ValueError):
        parse_run_event_payload({"version": "v2", "data": {}})
