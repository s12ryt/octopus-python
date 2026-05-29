from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from octopus_python.app import create_app
from octopus_python.config import AppConfig, DatabaseConfig, LogConfig, ServerConfig
from octopus_python.database import APIKey, Channel, RelayLog, StatsHourly, close_db, init_db, session_scope
from octopus_python.relay import merge_usage
from octopus_python.schemas import ChannelCreateRequest, ChannelUpdateRequest, GroupCreateRequest, SettingRequest
from octopus_python.services import (
    add_relay_log,
    cleanup_old_relay_logs,
    create_channel,
    create_group,
    get_stats_hourly,
    init_services,
    issue_stream_token,
    list_relay_logs,
    record_usage,
    set_setting,
    today_str,
    update_channel,
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
