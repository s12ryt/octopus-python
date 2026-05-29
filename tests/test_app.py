from __future__ import annotations

import asyncio
import time
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from octopus_python.app import create_app
from octopus_python.config import AppConfig, DatabaseConfig, LogConfig, ServerConfig
from octopus_python.database import APIKey, Channel, LLMInfo, RelayLog, StatsHourly, close_db, init_db, session_scope
from octopus_python.relay import FORM_FIELDS_KEY, FORM_FILES_KEY, merge_usage, prepare_body, sanitize_request_body_for_log
from octopus_python.schemas import ChannelCreateRequest, ChannelUpdateRequest, GroupCreateRequest, SettingRequest
from octopus_python.services import (
    add_relay_log,
    cleanup_old_relay_logs,
    create_channel,
    create_group,
    get_stats_hourly,
    httpx_client_options,
    init_services,
    issue_stream_token,
    list_relay_logs,
    measure_base_url_delay,
    parse_models_dev_prices,
    record_usage,
    set_setting,
    today_str,
    update_channel,
    update_channel_base_url_delay,
    update_model_prices,
)


@pytest.fixture()
def client(tmp_path: Path):
    close_db()
    db_path = tmp_path / "octopus.db"
    init_db("sqlite", str(db_path))
    asyncio.run(init_services())
    config = AppConfig(server=ServerConfig(), log=LogConfig(), database=DatabaseConfig(type="sqlite", path=str(db_path)))
    with TestClient(create_app(config)) as test_client:
        yield test_client
    close_db()


def login(client: TestClient) -> str:
    response = client.post("/api/v1/user/login", json={"username": "admin", "password": "admin", "expire": 0})
    assert response.status_code == 200
    payload = response.json()["data"]
    assert payload["token"]
    assert payload["expire_at"]
    return payload["token"]


def test_user_status_and_change_messages_match_go(client: TestClient) -> None:
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}

    assert client.get("/api/v1/user/status", headers=headers).json()["data"] == "ok"
    response = client.post(
        "/api/v1/user/change-password",
        headers=headers,
        json={"old_password": "admin", "new_password": "admin"},
    )
    assert response.json()["data"] == "password changed successfully"


def test_log_list_returns_frontend_compatible_array(client: TestClient) -> None:
    token = login(client)
    add_relay_log(
        {
            "time": int(time.time()),
            "request_model_name": "gpt-test",
            "request_api_key_name": "default",
            "channel": 1,
            "channel_name": "openai",
            "actual_model_name": "gpt-test",
            "input_tokens": 1,
            "output_tokens": 2,
            "ftut": 0,
            "use_time": 3,
            "cost": 0.0,
            "request_content": "{}",
            "response_content": "{}",
            "error": "",
            "attempts": [],
            "total_attempts": 1,
        }
    )

    direct = list_relay_logs(page=1, page_size=20)
    assert isinstance(direct, list)
    assert direct[0]["request_model_name"] == "gpt-test"

    response = client.get("/api/v1/log/list?page=1&page_size=20", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 200
    data = response.json()["data"]
    assert isinstance(data, list)
    assert data[0]["request_model_name"] == "gpt-test"


def test_hourly_stats_use_today_hour_buckets(client: TestClient) -> None:
    with session_scope() as session:
        record_usage(
            session,
            api_key_id=0,
            channel_id=0,
            actual_model="gpt-test",
            prompt_tokens=10,
            completion_tokens=5,
            wait_time=42,
            success=True,
        )

    rows = get_stats_hourly()
    assert rows
    assert rows[-1]["date"] == today_str()
    assert 0 <= rows[-1]["hour"] <= 23
    assert rows[-1]["input_token"] == 10

    with session_scope() as session:
        stored = session.get(StatsHourly, rows[-1]["hour"])
        assert stored is not None
        assert stored.date == today_str()


def test_update_channel_preserves_auto_group_zero(client: TestClient) -> None:
    channel = create_channel(
        ChannelCreateRequest(
            name="test-channel",
            type="openai/chat_completions",
            enabled=True,
            base_urls=[{"url": "https://example.com/v1", "delay": 0}],
            keys=[],
            model="gpt-test",
            custom_model="",
            proxy=False,
            auto_sync=False,
            auto_group=1,
            custom_header=[],
            param_override="",
            channel_proxy="",
            match_regex="",
        )
    )
    updated = update_channel(ChannelUpdateRequest(id=channel["id"], auto_group=0))
    assert updated["auto_group"] == 0
    with session_scope() as session:
        assert session.get(Channel, channel["id"]).auto_group == 0


def test_auto_group_matches_existing_groups_like_go(client: TestClient) -> None:
    create_group(GroupCreateRequest(name="gpt-4o", mode=1))
    create_group(GroupCreateRequest(name="claude", mode=1))
    create_group(GroupCreateRequest(name="regex", mode=1, match_regex=r"^gemini-.*flash$"))

    create_channel(
        ChannelCreateRequest(
            name="exact-channel",
            type="openai/chat_completions",
            base_urls=[{"url": "https://example.com/v1", "delay": 0}],
            model="gpt-4o,gpt-4o-mini",
            auto_group=2,
        )
    )
    create_channel(
        ChannelCreateRequest(
            name="fuzzy-channel",
            type="anthropic/messages",
            base_urls=[{"url": "https://example.com/v1", "delay": 0}],
            model="claude-3-5-sonnet,other",
            auto_group=1,
        )
    )
    create_channel(
        ChannelCreateRequest(
            name="regex-channel",
            type="gemini/contents",
            base_urls=[{"url": "https://example.com/v1beta", "delay": 0}],
            model="gemini-2.5-flash,gemini-2.5-pro",
            auto_group=3,
        )
    )

    groups = {group["name"]: group for group in client.get("/api/v1/group/list", headers={"Authorization": f"Bearer {login(client)}"}).json()["data"]}
    assert [item["model_name"] for item in groups["gpt-4o"]["items"]] == ["gpt-4o"]
    assert [item["model_name"] for item in groups["claude"]["items"]] == ["claude-3-5-sonnet"]
    assert [item["model_name"] for item in groups["regex"]["items"]] == ["gemini-2.5-flash"]


def test_stream_token_is_one_time_and_model_list_auth(client: TestClient) -> None:
    token = login(client)
    headers = {"Authorization": f"Bearer {token}"}
    stream_token = client.get("/api/v1/log/stream-token", headers=headers).json()["data"]["token"]
    assert stream_token
    assert issue_stream_token() != stream_token

    # API key auth smoke test for /v1/models.
    with session_scope() as session:
        api_key = APIKey(name="k", api_key="sk-octopus-testkey", enabled=True)
        session.add(api_key)
    response = client.get("/v1/models", headers={"Authorization": "Bearer sk-octopus-testkey"})
    assert response.status_code == 200
    assert response.json()["object"] == "list"


def test_stream_usage_parser_supports_openai_and_anthropic_chunks(client: TestClient) -> None:
    usage = merge_usage((0, 0), b'data: {"usage":{"prompt_tokens":3,"completion_tokens":5}}\n\n')
    assert usage == (3, 5)
    usage = merge_usage(
        usage,
        b'data: {"type":"message_start","message":{"usage":{"input_tokens":7}}}\n\n'
        b'data: {"type":"message_delta","usage":{"output_tokens":11}}\n\n',
    )
    assert usage == (7, 11)


def test_cleanup_old_relay_logs_honors_retention_setting(client: TestClient) -> None:
    set_setting(SettingRequest(key="relay_log_keep_period", value="1"))
    set_setting(SettingRequest(key="relay_log_keep_enabled", value="true"))
    with session_scope() as session:
        session.add(
            RelayLog(
                id=1,
                time=int(time.time()) - 3 * 86400,
                request_model_name="old",
                actual_model_name="old",
            )
        )
    add_relay_log({"time": int(time.time()), "request_model_name": "new", "actual_model_name": "new"})

    assert cleanup_old_relay_logs() == 1
    names = [row["request_model_name"] for row in list_relay_logs(page=1, page_size=20)]
    assert "old" not in names
    assert "new" in names


def test_parse_models_dev_prices_filters_supported_providers() -> None:
    prices = parse_models_dev_prices(
        {
            "openai": {
                "models": {
                    "gpt-test": {
                        "id": "GPT-Test",
                        "cost": {"input": 1.25, "output": "2.5", "cache_read": 0.5, "cache_write": None},
                    }
                }
            },
            "unsupported": {"models": {"x": {"id": "x", "cost": {"input": 99}}}},
        }
    )
    assert prices == {"gpt-test": {"input": 1.25, "output": 2.5, "cache_read": 0.5, "cache_write": 0.0}}


def test_update_model_prices_uses_models_dev_without_overwriting_custom_price(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_fetch_prices() -> dict[str, dict[str, float]]:
        return {
            "gpt-priced": {"input": 1.0, "output": 2.0, "cache_read": 0.1, "cache_write": 0.2},
            "custom-priced": {"input": 9.0, "output": 9.0, "cache_read": 9.0, "cache_write": 9.0},
        }

    monkeypatch.setattr("octopus_python.services.fetch_models_dev_prices", fake_fetch_prices)
    create_channel(
        ChannelCreateRequest(
            name="priced",
            type="openai/chat_completions",
            model="gpt-priced",
            keys=[{"channel_key": "sk-test", "enabled": True}],
        )
    )
    with session_scope() as session:
        session.add(LLMInfo(name="custom-priced", input=3.0, output=4.0, cache_read=0.0, cache_write=0.0))

    asyncio.run(update_model_prices())

    with session_scope() as session:
        auto_price = session.get(LLMInfo, "gpt-priced")
        custom_price = session.get(LLMInfo, "custom-priced")
        assert auto_price is not None
        assert (auto_price.input, auto_price.output, auto_price.cache_read, auto_price.cache_write) == (1.0, 2.0, 0.1, 0.2)
        assert custom_price is not None
        assert (custom_price.input, custom_price.output, custom_price.cache_read, custom_price.cache_write) == (
            3.0,
            4.0,
            0.0,
            0.0,
        )


def test_multipart_image_body_preserves_files_but_sanitizes_logs() -> None:
    body = {
        "model": "gpt-image-1",
        "prompt": "edit this",
        FORM_FIELDS_KEY: [("model", "gpt-image-1"), ("prompt", "edit this")],
        FORM_FILES_KEY: [
            {
                "field": "image",
                "filename": "input.png",
                "content": b"binary-image",
                "content_type": "image/png",
            }
        ],
    }

    prepared, converted = prepare_body("openai/images_edits", "images_edits", body, "actual-image-model")
    assert not converted
    assert prepared["model"] == "actual-image-model"
    assert ("model", "actual-image-model") in prepared[FORM_FIELDS_KEY]
    assert prepared[FORM_FILES_KEY][0]["content"] == b"binary-image"

    safe = sanitize_request_body_for_log(prepared)
    assert FORM_FILES_KEY not in safe
    assert safe["files"] == [
        {"field": "image", "filename": "input.png", "content_type": "image/png", "size": len(b"binary-image")}
    ]


def test_httpx_client_options_matches_go_proxy_rules(client: TestClient) -> None:
    set_setting(SettingRequest(key="proxy_url", value="http://127.0.0.1:8888"))
    assert httpx_client_options(proxy_enabled=False) == {"trust_env": False}
    assert httpx_client_options(proxy_enabled=True, channel_proxy="http://127.0.0.1:9999") == {
        "proxy": "http://127.0.0.1:9999",
        "trust_env": False,
    }
    assert httpx_client_options(proxy_enabled=True) == {"proxy": "http://127.0.0.1:8888", "trust_env": False}


def test_base_url_delay_update_persists_measured_delay(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    channel = create_channel(
        ChannelCreateRequest(
            name="delay-channel",
            type="openai/chat_completions",
            base_urls=[{"url": "https://example.com/v1", "delay": 0}],
            model="gpt-delay",
        )
    )

    async def fake_measure(url: str, client_options: dict[str, object] | None = None) -> int:
        assert url == "https://example.com/v1"
        assert client_options == {"trust_env": False}
        return 123

    monkeypatch.setattr("octopus_python.services.measure_base_url_delay", fake_measure)
    asyncio.run(update_channel_base_url_delay(channel["id"]))

    with session_scope() as session:
        stored = session.get(Channel, channel["id"])
        assert stored is not None
        assert stored.base_urls[0]["delay"] == 123


def test_measure_base_url_delay_uses_httpx_transport() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="ok")

    transport = httpx.MockTransport(handler)
    delay = asyncio.run(measure_base_url_delay("https://example.com", {"transport": transport, "trust_env": False}))
    assert delay >= 1
