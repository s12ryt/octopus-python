from __future__ import annotations

import asyncio
import random
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

import httpx
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from . import VERSION
from .database import (
    APIKey,
    Channel,
    ChannelKey,
    Group,
    GroupItem,
    LLMInfo,
    RelayLog,
    Setting,
    StatsAPIKey,
    StatsChannel,
    StatsDaily,
    StatsHourly,
    StatsModel,
    StatsTotal,
    User,
    session_scope,
)
from .schemas import (
    APIKeyCreateRequest,
    APIKeyUpdateRequest,
    ChannelCreateRequest,
    ChannelUpdateRequest,
    FetchModelRequest,
    GroupCreateRequest,
    GroupUpdateRequest,
    LLMInfoRequest,
    SettingRequest,
    split_csv,
)
from .security import generate_api_key, generate_stream_token, hash_password, verify_password

SETTING_DEFAULTS: dict[str, str] = {
    "proxy_url": "",
    "stats_save_interval": "10",
    "model_info_update_interval": "24",
    "sync_llm_interval": "24",
    "relay_log_keep_period": "7",
    "relay_log_keep_enabled": "true",
    "cors_allow_origins": "",
    "circuit_breaker_threshold": "5",
    "circuit_breaker_cooldown": "60",
    "circuit_breaker_max_cooldown": "600",
}

NUMERIC_SETTINGS = {
    "stats_save_interval",
    "model_info_update_interval",
    "sync_llm_interval",
    "relay_log_keep_period",
    "circuit_breaker_threshold",
    "circuit_breaker_cooldown",
    "circuit_breaker_max_cooldown",
}

SENSITIVE_HEADERS = {"authorization", "x-api-key", "api-key", "anthropic-api-key", "openai-api-key"}
MODELS_DEV_PRICE_URL = "https://models.dev/api.json"
MODELS_DEV_PROVIDERS = {
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "xai",
    "alibaba",
    "zhipuai",
    "minimax",
    "moonshotai",
    "v0",
}
CHANNEL_PATHS = {
    "openai/chat_completions": "/chat/completions",
    "openai/responses": "/responses",
    "openai/embeddings": "/embeddings",
    "anthropic/messages": "/messages",
    "gemini/contents": "",
    "doubao": "/chat/completions",
}
BASE_URL_DELAY_INTERVAL_SECONDS = 3600

last_model_update_time = ""
last_channel_sync_time = ""
relay_stream_tokens: set[str] = set()
relay_subscribers: set[asyncio.Queue[dict[str, Any]]] = set()
round_robin_counters: defaultdict[int, int] = defaultdict(int)
sticky_sessions: dict[str, tuple[int, int, float]] = {}
circuit_breakers: dict[str, dict[str, Any]] = {}
background_tasks: set[asyncio.Task[Any]] = set()
background_stop_event: asyncio.Event | None = None
llm_price_cache: dict[str, dict[str, float]] = {}


def _normalize_base_url(base_url: str, suffix: str = "v1") -> str:
    """Mirror the Go helper/axonhub base URL behavior for common provider endpoints."""
    base = (base_url or "").rstrip("/")
    suffix = suffix.strip("/")
    if not suffix:
        return base
    if base.endswith("/" + suffix) or base.endswith("/v1") or base.endswith("/v1beta") or base.endswith("/v3"):
        return base
    return f"{base}/{suffix}"


def _diff_models(old: Iterable[str], new: Iterable[str]) -> tuple[list[str], list[str]]:
    old_norm = [x.strip() for x in old if x and x.strip()]
    new_norm = [x.strip() for x in new if x and x.strip()]
    old_set = set(old_norm)
    new_set = set(new_norm)
    return [x for x in old_norm if x not in new_set], [x for x in new_norm if x not in old_set]


def now_ms() -> int:
    return int(time.time() * 1000)


def today_str() -> str:
    return datetime.now().strftime("%Y%m%d")


def current_hour() -> int:
    return datetime.now().hour


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def init_services() -> None:
    with session_scope() as session:
        ensure_settings(session)
        ensure_admin_user(session)
        ensure_total_stats(session)


async def shutdown_services() -> None:
    await stop_background_tasks()


def save_runtime_state() -> None:
    return None


async def start_background_tasks() -> None:
    """Start lightweight Go-like maintenance loops on the running ASGI event loop."""
    global background_stop_event
    if background_tasks:
        return
    background_stop_event = asyncio.Event()
    for coro in [
        _periodic_model_price_sync(),
        _periodic_channel_sync(),
        _periodic_base_url_delay_update(),
        _periodic_relay_log_cleanup(),
    ]:
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)


async def stop_background_tasks() -> None:
    global background_stop_event
    if background_stop_event is not None:
        background_stop_event.set()
    tasks = list(background_tasks)
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    background_tasks.clear()
    background_stop_event = None


async def _sleep_or_stop(seconds: float) -> bool:
    event = background_stop_event
    if event is None:
        return True
    try:
        await asyncio.wait_for(event.wait(), timeout=max(seconds, 0.1))
        return True
    except TimeoutError:
        return False


async def _periodic_model_price_sync() -> None:
    first_run = True
    while True:
        try:
            with session_scope() as session:
                interval_hours = max(get_setting_int(session, "model_info_update_interval", 24), 1)
            if not first_run and await _sleep_or_stop(interval_hours * 3600):
                return
            first_run = False
            await update_model_prices()
        except asyncio.CancelledError:
            raise
        except Exception:
            await _sleep_or_stop(60)


async def _periodic_channel_sync() -> None:
    first_run = True
    while True:
        try:
            with session_scope() as session:
                interval_hours = max(get_setting_int(session, "sync_llm_interval", 24), 1)
            if not first_run and await _sleep_or_stop(interval_hours * 3600):
                return
            first_run = False
            await sync_channels()
        except asyncio.CancelledError:
            raise
        except Exception:
            await _sleep_or_stop(60)


async def _periodic_relay_log_cleanup() -> None:
    while True:
        try:
            if await _sleep_or_stop(3600):
                return
            cleanup_old_relay_logs()
        except asyncio.CancelledError:
            raise
        except Exception:
            await _sleep_or_stop(60)


async def _periodic_base_url_delay_update() -> None:
    first_run = True
    while True:
        try:
            if not first_run and await _sleep_or_stop(BASE_URL_DELAY_INTERVAL_SECONDS):
                return
            first_run = False
            await update_all_channel_base_url_delays()
        except asyncio.CancelledError:
            raise
        except Exception:
            await _sleep_or_stop(60)


def ensure_settings(session: Session) -> None:
    for key, value in SETTING_DEFAULTS.items():
        if session.get(Setting, key) is None:
            session.add(Setting(key=key, value=value))


def ensure_admin_user(session: Session) -> None:
    exists = session.scalar(select(User.id).limit(1))
    if not exists:
        session.add(User(username="admin", password=hash_password("admin")))


def ensure_total_stats(session: Session) -> StatsTotal:
    stats = session.get(StatsTotal, 1)
    if stats is None:
        stats = StatsTotal(id=1)
        session.add(stats)
        session.flush()
    return stats


def get_setting(session: Session, key: str, default: str = "") -> str:
    setting = session.get(Setting, key)
    if setting is None:
        default = SETTING_DEFAULTS.get(key, default)
        setting = Setting(key=key, value=default)
        session.add(setting)
        session.flush()
    return setting.value


def get_setting_int(session: Session, key: str, default: int) -> int:
    try:
        return int(get_setting(session, key, str(default)))
    except Exception:
        return default


def validate_setting(req: SettingRequest) -> None:
    if req.key in NUMERIC_SETTINGS:
        int(req.value)
    if req.key == "relay_log_keep_enabled" and req.value.lower() not in {"true", "false"}:
        raise ValueError("relay_log_keep_enabled must be true or false")
    if req.key == "proxy_url" and req.value:
        if not re.match(r"^(https?|socks5)://[^\s/$.?#].[^\s]*$", req.value):
            raise ValueError("proxy_url must be empty or valid http/https/socks5 URL")


def httpx_client_options(
    *,
    proxy_enabled: bool,
    channel_proxy: str = "",
    session: Session | None = None,
) -> dict[str, Any]:
    """Mirror Go ChannelHttpClient proxy selection for httpx clients.

    proxy=false disables environment proxy. proxy=true uses channel_proxy first,
    then the global proxy_url setting, otherwise httpx environment proxy.
    """
    if not proxy_enabled:
        return {"trust_env": False}
    proxy_url = (channel_proxy or "").strip()
    if not proxy_url:
        if session is not None:
            proxy_url = get_setting(session, "proxy_url", "").strip()
        else:
            with session_scope() as local_session:
                proxy_url = get_setting(local_session, "proxy_url", "").strip()
    if proxy_url:
        return {"proxy": proxy_url, "trust_env": False}
    return {"trust_env": True}


def list_settings() -> list[dict[str, Any]]:
    with session_scope() as session:
        ensure_settings(session)
        return [s.to_dict(False) for s in session.scalars(select(Setting).order_by(Setting.key)).all()]


def set_setting(req: SettingRequest) -> dict[str, Any]:
    validate_setting(req)
    with session_scope() as session:
        setting = session.get(Setting, req.key)
        if setting is None:
            setting = Setting(key=req.key, value=req.value)
            session.add(setting)
        else:
            setting.value = req.value
        return setting.to_dict(False)


def get_user() -> User | None:
    with session_scope() as session:
        user = session.scalar(select(User).order_by(User.id).limit(1))
        if user:
            session.expunge(user)
        return user


def verify_user(username: str, password: str) -> User | None:
    with session_scope() as session:
        user = session.scalar(select(User).where(User.username == username).limit(1))
        if user and verify_password(password, user.password):
            session.expunge(user)
            return user
        return None


def change_password(old_password: str, new_password: str) -> None:
    with session_scope() as session:
        user = session.scalar(select(User).order_by(User.id).limit(1))
        if not user or not verify_password(old_password, user.password):
            raise ValueError("old password is incorrect")
        user.password = hash_password(new_password)


def change_username(new_username: str) -> None:
    with session_scope() as session:
        user = session.scalar(select(User).order_by(User.id).limit(1))
        if not user:
            raise ValueError("user not found")
        if user.username == new_username:
            raise ValueError("new username is same as old username")
        user.username = new_username


def _api_key_cost(session: Session, api_key_id: int) -> float:
    stats = session.get(StatsAPIKey, api_key_id)
    if not stats:
        return 0.0
    return (stats.input_cost or 0) + (stats.output_cost or 0)


def list_api_keys() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [k.to_dict(False) for k in session.scalars(select(APIKey).order_by(APIKey.id)).all()]


def create_api_key(req: APIKeyCreateRequest) -> dict[str, Any]:
    with session_scope() as session:
        item = APIKey(
            name=req.name,
            api_key=generate_api_key(),
            enabled=req.enabled,
            expire_at=req.expire_at or 0,
            max_cost=req.max_cost or 0.0,
            supported_models=req.supported_models or "",
        )
        session.add(item)
        session.flush()
        return item.to_dict(False)


def update_api_key(req: APIKeyUpdateRequest) -> dict[str, Any]:
    with session_scope() as session:
        item = session.get(APIKey, req.id)
        if item is None:
            raise LookupError("api key not found")
        item.name = req.name
        item.enabled = req.enabled
        item.expire_at = req.expire_at or 0
        item.max_cost = req.max_cost or 0.0
        item.supported_models = req.supported_models or ""
        return item.to_dict(False)


def delete_api_key(api_key_id: int) -> None:
    with session_scope() as session:
        item = session.get(APIKey, api_key_id)
        if item is None:
            raise LookupError("api key not found")
        stats = session.get(StatsAPIKey, api_key_id)
        if stats:
            session.delete(stats)
        session.delete(item)


def get_api_key_by_key(api_key: str) -> dict[str, Any] | None:
    with session_scope() as session:
        item = session.scalar(select(APIKey).where(APIKey.api_key == api_key).limit(1))
        if item is None:
            return None
        data = item.to_dict(False)
        data["used_cost"] = _api_key_cost(session, item.id)
        return data


def get_api_key_stats(api_key_id: int) -> dict[str, Any]:
    with session_scope() as session:
        stats = session.get(StatsAPIKey, api_key_id)
        if stats is None:
            stats = StatsAPIKey(api_key_id=api_key_id)
            session.add(stats)
            session.flush()
        info = session.get(APIKey, api_key_id)
        if info is None:
            raise LookupError("api key not found")
        info_data = info.to_dict(False)
        if not info_data.get("supported_models"):
            info_data["supported_models"] = ", ".join(list_group_model_names(session))
        return {"stats": stats.to_dict(False), "info": info_data}


def normalize_channel_dict(channel: Channel) -> dict[str, Any]:
    data = channel.to_dict(True)
    data["base_urls"] = data.get("base_urls") or []
    data["custom_header"] = data.get("custom_header") or []
    data["keys"] = data.get("keys") or []
    if not data.get("stats"):
        data["stats"] = zero_stats({"channel_id": channel.id})
    return data


def zero_stats(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {
        "input_token": 0,
        "output_token": 0,
        "input_cost": 0.0,
        "output_cost": 0.0,
        "wait_time": 0,
        "request_success": 0,
        "request_failed": 0,
    }
    if extra:
        data.update(extra)
    return data


def list_channels() -> list[dict[str, Any]]:
    with session_scope() as session:
        rows = session.scalars(
            select(Channel).options(selectinload(Channel.keys), selectinload(Channel.stats)).order_by(Channel.id)
        ).all()
        for ch in rows:
            if ch.stats is None:
                ch.stats = StatsChannel(channel_id=ch.id)
                session.add(ch.stats)
                session.flush()
        return [normalize_channel_dict(ch) for ch in rows]


def _model_names_from_channel(channel: Channel | dict[str, Any]) -> list[str]:
    if isinstance(channel, dict):
        model = channel.get("model") or ""
        custom = channel.get("custom_model") or ""
    else:
        model = channel.model or ""
        custom = channel.custom_model or ""
    return split_csv(model) + [m for m in split_csv(custom) if m not in split_csv(model)]


def create_missing_prices(session: Session, names: Iterable[str]) -> None:
    for name in names:
        key = name.lower()
        if not key:
            continue
        if session.get(LLMInfo, key) is None:
            price = llm_price_cache.get(key) or {}
            session.add(
                LLMInfo(
                    name=key,
                    input=float(price.get("input") or 0),
                    output=float(price.get("output") or 0),
                    cache_read=float(price.get("cache_read") or 0),
                    cache_write=float(price.get("cache_write") or 0),
                )
            )


def auto_group_channel(session: Session, channel: Channel) -> None:
    if not channel.auto_group:
        return
    models = _model_names_from_channel(channel)
    if not models:
        return

    # Go 版 AutoGroup 不會自行建立 group；而是依既有 group 名稱/regex 將 channel model 掛入。
    groups = session.scalars(select(Group).order_by(Group.id)).all()
    for group in groups:
        matched: list[str] = []
        if channel.auto_group == 2:  # exact
            matched = [m for m in models if m.lower() == group.name.lower()]
        elif channel.auto_group == 1:  # fuzzy
            group_name = group.name.strip().lower()
            if group_name:
                matched = [m for m in models if group_name in m.lower()]
        elif channel.auto_group == 3:  # regex, fallback to exact when regex is empty
            if group.match_regex:
                try:
                    pattern = re.compile(group.match_regex)
                except re.error:
                    continue
                matched = [m for m in models if pattern.search(m)]
            else:
                matched = [m for m in models if m.lower() == group.name.lower()]
        for model_name in matched:
            exists = session.scalar(
                select(GroupItem.id)
                .where(GroupItem.group_id == group.id, GroupItem.channel_id == channel.id, GroupItem.model_name == model_name)
                .limit(1)
            )
            if not exists:
                session.add(GroupItem(group_id=group.id, channel_id=channel.id, model_name=model_name, priority=0, weight=1))


def create_channel(req: ChannelCreateRequest) -> dict[str, Any]:
    with session_scope() as session:
        channel = Channel(
            name=req.name,
            type=req.type,
            enabled=req.enabled,
            base_urls=[x.model_dump() for x in req.base_urls],
            model=req.model,
            custom_model=req.custom_model or "",
            proxy=req.proxy,
            auto_sync=req.auto_sync,
            auto_group=req.auto_group,
            custom_header=[x.model_dump() for x in req.custom_header],
            param_override=req.param_override or "",
            channel_proxy=req.channel_proxy or "",
            match_regex=req.match_regex or "",
        )
        session.add(channel)
        session.flush()
        for key in req.keys:
            session.add(ChannelKey(channel_id=channel.id, enabled=key.enabled, channel_key=key.channel_key, remark=key.remark))
        session.add(StatsChannel(channel_id=channel.id))
        create_missing_prices(session, _model_names_from_channel(channel))
        auto_group_channel(session, channel)
        session.flush()
        session.refresh(channel)
        return normalize_channel_dict(channel)


def update_channel(req: ChannelUpdateRequest) -> dict[str, Any]:
    with session_scope() as session:
        channel = session.get(Channel, req.id, options=[selectinload(Channel.keys), selectinload(Channel.stats)])
        if channel is None:
            raise LookupError("channel not found")
        fields = [
            "name",
            "type",
            "enabled",
            "model",
            "custom_model",
            "proxy",
            "auto_sync",
            "auto_group",
            "param_override",
            "channel_proxy",
            "match_regex",
        ]
        updates = req.model_dump(exclude_unset=True)
        for field in fields:
            if field not in updates:
                continue
            value = updates[field]
            if value is None:
                value = False if isinstance(getattr(channel, field), bool) else ""
            setattr(channel, field, value)
        if req.base_urls is not None:
            channel.base_urls = [x.model_dump() for x in req.base_urls]
        if req.custom_header is not None:
            channel.custom_header = [x.model_dump() for x in req.custom_header]
        for key_id in req.keys_to_delete:
            key = session.get(ChannelKey, key_id)
            if key and key.channel_id == channel.id:
                session.delete(key)
        for key_req in req.keys_to_update:
            key = session.get(ChannelKey, key_req.id)
            if key and key.channel_id == channel.id:
                if key_req.enabled is not None:
                    key.enabled = key_req.enabled
                if key_req.channel_key is not None:
                    key.channel_key = key_req.channel_key
                if key_req.remark is not None:
                    key.remark = key_req.remark
        for key_req in req.keys_to_add:
            session.add(ChannelKey(channel_id=channel.id, enabled=key_req.enabled, channel_key=key_req.channel_key, remark=key_req.remark))
        create_missing_prices(session, _model_names_from_channel(channel))
        auto_group_channel(session, channel)
        session.flush()
        session.refresh(channel)
        return normalize_channel_dict(channel)


def enable_channel(channel_id: int, enabled: bool) -> None:
    with session_scope() as session:
        channel = session.get(Channel, channel_id)
        if channel is None:
            raise LookupError("channel not found")
        channel.enabled = enabled


def delete_channel(channel_id: int) -> None:
    with session_scope() as session:
        channel = session.get(Channel, channel_id)
        if channel is None:
            raise LookupError("channel not found")
        session.execute(delete(GroupItem).where(GroupItem.channel_id == channel_id))
        stats = session.get(StatsChannel, channel_id)
        if stats:
            session.delete(stats)
        session.delete(channel)


def choose_base_url(channel: Channel | dict[str, Any]) -> str:
    base_urls = channel["base_urls"] if isinstance(channel, dict) else channel.base_urls
    base_urls = base_urls or []
    candidates = [u for u in base_urls if isinstance(u, dict) and u.get("url")]
    if not candidates:
        return ""
    return min(candidates, key=lambda x: x.get("delay", 0) or 0)["url"].rstrip("/")


async def measure_base_url_delay(url: str, client_options: dict[str, Any] | None = None) -> int:
    target = (url or "").rstrip("/")
    if not target:
        return 0
    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0), **(client_options or {})) as client:
            try:
                response = await client.head(target)
                if response.status_code == 405:
                    response = await client.get(target)
            except httpx.HTTPError:
                response = await client.get(target)
            await response.aread()
        return max(int((time.perf_counter() - start) * 1000), 1)
    except Exception:
        return 0


async def update_channel_base_url_delay(channel_id: int) -> None:
    with session_scope() as session:
        channel = session.get(Channel, channel_id)
        if channel is None:
            return
        base_urls = [dict(item) for item in (channel.base_urls or []) if isinstance(item, dict)]
        client_options = httpx_client_options(
            proxy_enabled=bool(channel.proxy), channel_proxy=channel.channel_proxy or "", session=session
        )
    changed = False
    for item in base_urls:
        if not item.get("url"):
            continue
        delay = await measure_base_url_delay(str(item["url"]), client_options)
        if delay and item.get("delay") != delay:
            item["delay"] = delay
            changed = True
    if changed:
        with session_scope() as session:
            channel = session.get(Channel, channel_id)
            if channel is not None:
                channel.base_urls = base_urls


async def update_all_channel_base_url_delays() -> None:
    with session_scope() as session:
        channel_ids = list(session.scalars(select(Channel.id).order_by(Channel.id)).all())
    for channel_id in channel_ids:
        await update_channel_base_url_delay(channel_id)


def choose_channel_key(channel: Channel) -> ChannelKey | None:
    now = int(time.time())
    keys = [k for k in channel.keys if k.enabled and k.channel_key]
    keys = [k for k in keys if not (k.status_code == 429 and now - (k.last_use_time_stamp or 0) < 300)]
    if not keys:
        return None
    return min(keys, key=lambda k: k.total_cost or 0)


def list_group_model_names(session: Session) -> list[str]:
    return [x for x in session.scalars(select(Group.name).order_by(Group.name)).all()]


def validate_group_regex(pattern: str | None) -> None:
    if pattern:
        re.compile(pattern)


def list_groups() -> list[dict[str, Any]]:
    with session_scope() as session:
        groups = session.scalars(select(Group).options(selectinload(Group.items)).order_by(Group.id)).all()
        return [g.to_dict(True) for g in groups]


def create_group(req: GroupCreateRequest) -> dict[str, Any]:
    validate_group_regex(req.match_regex)
    with session_scope() as session:
        group = Group(
            name=req.name,
            mode=req.mode,
            match_regex=req.match_regex or "",
            first_token_time_out=req.first_token_time_out or 0,
            session_keep_time=req.session_keep_time or 0,
        )
        session.add(group)
        session.flush()
        for item in req.items:
            session.add(
                GroupItem(
                    group_id=group.id,
                    channel_id=item.channel_id,
                    model_name=item.model_name,
                    priority=item.priority,
                    weight=item.weight,
                )
            )
        session.flush()
        session.refresh(group)
        return group.to_dict(True)


def update_group(req: GroupUpdateRequest) -> dict[str, Any]:
    if req.match_regex is not None:
        validate_group_regex(req.match_regex)
    with session_scope() as session:
        group = session.get(Group, req.id, options=[selectinload(Group.items)])
        if group is None:
            raise LookupError("group not found")
        updates = req.model_dump(exclude_unset=True)
        for field in ["name", "mode", "match_regex", "first_token_time_out", "session_keep_time"]:
            if field in updates and updates[field] is not None:
                setattr(group, field, updates[field])
        for item_id in req.items_to_delete:
            item = session.get(GroupItem, item_id)
            if item and item.group_id == group.id:
                session.delete(item)
        for item_req in req.items_to_update:
            item = session.get(GroupItem, item_req.id)
            if item and item.group_id == group.id:
                item.priority = item_req.priority
                item.weight = item_req.weight
        for item_req in req.items_to_add:
            exists = session.scalar(
                select(GroupItem.id)
                .where(
                    GroupItem.group_id == group.id,
                    GroupItem.channel_id == item_req.channel_id,
                    GroupItem.model_name == item_req.model_name,
                )
                .limit(1)
            )
            if not exists:
                session.add(
                    GroupItem(
                        group_id=group.id,
                        channel_id=item_req.channel_id,
                        model_name=item_req.model_name,
                        priority=item_req.priority,
                        weight=item_req.weight,
                    )
                )
        session.flush()
        session.refresh(group)
        return group.to_dict(True)


def delete_group(group_id: int) -> None:
    with session_scope() as session:
        group = session.get(Group, group_id)
        if group is None:
            raise LookupError("group not found")
        session.delete(group)


def list_llm_infos() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [m.to_dict(False) for m in session.scalars(select(LLMInfo).order_by(LLMInfo.name)).all()]


def create_llm_info(req: LLMInfoRequest) -> dict[str, Any]:
    with session_scope() as session:
        name = req.name.lower()
        if session.get(LLMInfo, name) is not None:
            raise FileExistsError("model exists")
        item = LLMInfo(name=name, input=req.input, output=req.output, cache_read=req.cache_read, cache_write=req.cache_write)
        session.add(item)
        session.flush()
        return item.to_dict(False)


def update_llm_info(req: LLMInfoRequest) -> dict[str, Any]:
    with session_scope() as session:
        name = req.name.lower()
        item = session.get(LLMInfo, name)
        if item is None:
            item = LLMInfo(name=name)
            session.add(item)
        item.input = req.input
        item.output = req.output
        item.cache_read = req.cache_read
        item.cache_write = req.cache_write
        return item.to_dict(False)


def delete_llm_info(name: str) -> None:
    with session_scope() as session:
        item = session.get(LLMInfo, name.lower())
        if item is None:
            raise LookupError("model not found")
        session.delete(item)


def list_llm_channels() -> list[dict[str, Any]]:
    with session_scope() as session:
        channels = session.scalars(select(Channel).order_by(Channel.id)).all()
        out: list[dict[str, Any]] = []
        for ch in channels:
            for name in _model_names_from_channel(ch):
                out.append({"name": name, "enabled": ch.enabled, "channel_id": ch.id, "channel_name": ch.name})
        return out


def _float_price(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def parse_models_dev_prices(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    prices: dict[str, dict[str, float]] = {}
    for provider in MODELS_DEV_PROVIDERS:
        provider_data = payload.get(provider)
        if not isinstance(provider_data, dict):
            continue
        models = provider_data.get("models")
        if not isinstance(models, dict):
            continue
        for fallback_name, raw_model in models.items():
            if not isinstance(raw_model, dict):
                continue
            name = str(raw_model.get("id") or fallback_name).strip().lower()
            cost = raw_model.get("cost")
            if not name or not isinstance(cost, dict):
                continue
            prices[name] = {
                "input": _float_price(cost.get("input")),
                "output": _float_price(cost.get("output")),
                "cache_read": _float_price(cost.get("cache_read")),
                "cache_write": _float_price(cost.get("cache_write")),
            }
    return prices


async def fetch_models_dev_prices() -> dict[str, dict[str, float]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Octopus-Python/1.0"
    }
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.get(MODELS_DEV_PRICE_URL, headers=headers)
        response.raise_for_status()
        return parse_models_dev_prices(response.json())


def _has_nonzero_price(item: LLMInfo) -> bool:
    return bool((item.input or 0) or (item.output or 0) or (item.cache_read or 0) or (item.cache_write or 0))


def _apply_cached_price(item: LLMInfo, price: dict[str, float]) -> None:
    item.input = float(price.get("input") or 0)
    item.output = float(price.get("output") or 0)
    item.cache_read = float(price.get("cache_read") or 0)
    item.cache_write = float(price.get("cache_write") or 0)


async def update_model_prices() -> None:
    global last_model_update_time
    global llm_price_cache
    fetched_prices = await fetch_models_dev_prices()
    if fetched_prices:
        llm_price_cache = fetched_prices

    # Go 版價格來源優先順序：使用者在價格頁自訂 > models.dev 自動同步。
    # Python 版沒有單獨的 Go in-memory price map，因此把 models.dev 價格補進：
    # 1. 新增 channel model 時直接帶入價格。
    # 2. 既有全 0 價格列視為自動 placeholder，可被 models.dev 補齊。
    # 3. 既有非 0 價格列視為使用者自訂，不覆蓋。
    with session_scope() as session:
        channels = session.scalars(select(Channel)).all()
        for ch in channels:
            create_missing_prices(session, _model_names_from_channel(ch))
        for item in session.scalars(select(LLMInfo)).all():
            price = llm_price_cache.get(item.name.lower())
            if price and not _has_nonzero_price(item):
                _apply_cached_price(item, price)
    last_model_update_time = iso_now()


def get_last_model_update_time() -> str:
    return last_model_update_time


async def sync_channels() -> None:
    global last_channel_sync_time
    total_new_models: list[str] = []
    seen_total: set[str] = set()
    with session_scope() as session:
        channels = session.scalars(select(Channel).options(selectinload(Channel.keys)).order_by(Channel.id)).all()

    for channel in channels:
        if not channel.auto_sync:
            continue
        req = FetchModelRequest(
            type=channel.type,
            base_urls=channel.base_urls or [],
            keys=[{"enabled": k.enabled, "channel_key": k.channel_key, "remark": k.remark} for k in channel.keys],
            proxy=channel.proxy,
            channel_proxy=channel.channel_proxy or "",
            match_regex=channel.match_regex or "",
            custom_header=channel.custom_header or [],
        )
        fetched = await fetch_models(req)
        new_models = [m.strip().lower() for m in fetched if m and m.strip()]
        if not new_models:
            continue
        for model_name in new_models:
            if model_name not in seen_total:
                seen_total.add(model_name)
                total_new_models.append(model_name)
        old_models = split_csv(channel.model)
        deleted, added = _diff_models(old_models, new_models)
        if deleted or added:
            with session_scope() as session:
                stored = session.get(Channel, channel.id)
                if stored is not None:
                    stored.model = ",".join(new_models)
                    for deleted_model in deleted:
                        session.execute(
                            delete(GroupItem).where(
                                GroupItem.channel_id == channel.id,
                                GroupItem.model_name == deleted_model,
                            )
                        )
                    create_missing_prices(session, new_models)
                    auto_group_channel(session, stored)

    if total_new_models:
        with session_scope() as session:
            create_missing_prices(session, total_new_models)
    last_channel_sync_time = iso_now()


def get_last_channel_sync_time() -> str:
    return last_channel_sync_time


async def fetch_models(req: FetchModelRequest) -> list[str]:
    base = choose_base_url({"base_urls": [x.model_dump() for x in req.base_urls]})
    key = next((k.channel_key for k in req.keys if k.enabled and k.channel_key), "")
    if not base or not key:
        return []
    headers = {"Authorization": f"Bearer {key}"}
    for h in req.custom_header:
        if h.header_key and h.header_key.lower() not in SENSITIVE_HEADERS:
            headers[h.header_key] = h.header_value
    try:
        client_options = httpx_client_options(proxy_enabled=bool(req.proxy), channel_proxy=req.channel_proxy or "")
        async with httpx.AsyncClient(timeout=20, **client_options) as client:
            if req.type == "anthropic/messages":
                headers.pop("Authorization", None)
                headers["x-api-key"] = key
                headers["anthropic-version"] = "2023-06-01"
                names = await _fetch_anthropic_model_names(client, _normalize_base_url(base, "v1"), headers)
            elif req.type == "gemini/contents":
                headers.pop("Authorization", None)
                headers["x-goog-api-key"] = key
                gemini_base = _normalize_base_url(base, "v1beta")
                if base.rstrip("/").endswith("/v1"):
                    gemini_base = base.rstrip("/")
                names = await _fetch_gemini_model_names(client, gemini_base, headers)
            else:
                suffix = "v3" if req.type == "doubao" else "v1"
                names = await _fetch_openai_model_names(client, _normalize_base_url(base, suffix), headers)
            if not names and req.type in {"anthropic/messages", "gemini/contents"}:
                names = await _fetch_openai_model_names(client, _normalize_base_url(base, "v1"), headers)
            if req.match_regex:
                pattern = re.compile(req.match_regex)
                names = [name for name in names if pattern.search(name)]
            return sorted(set(names))
    except Exception:
        return []


async def _fetch_openai_model_names(client: httpx.AsyncClient, base: str, headers: dict[str, str]) -> list[str]:
    res = await client.get(base.rstrip("/") + "/models", headers=headers)
    data = res.json()
    models = data.get("data", data.get("models", [])) if isinstance(data, dict) else []
    names: list[str] = []
    for item in models:
        if isinstance(item, str):
            names.append(item)
        elif isinstance(item, dict):
            mid = item.get("id") or item.get("name") or item.get("model")
            if mid:
                names.append(str(mid))
    return names


async def _fetch_gemini_model_names(client: httpx.AsyncClient, base: str, headers: dict[str, str]) -> list[str]:
    names: list[str] = []
    page_token = ""
    while True:
        params = {"pageToken": page_token} if page_token else None
        res = await client.get(base.rstrip("/") + "/models", headers=headers, params=params)
        data = res.json()
        for item in data.get("models", []) if isinstance(data, dict) else []:
            if isinstance(item, dict):
                mid = item.get("name") or item.get("id")
                if mid:
                    names.append(str(mid).removeprefix("models/"))
        page_token = str(data.get("nextPageToken") or "") if isinstance(data, dict) else ""
        if not page_token:
            return names


async def _fetch_anthropic_model_names(client: httpx.AsyncClient, base: str, headers: dict[str, str]) -> list[str]:
    names: list[str] = []
    after_id = ""
    while True:
        params = {"after_id": after_id} if after_id else None
        res = await client.get(base.rstrip("/") + "/models", headers=headers, params=params)
        data = res.json()
        for item in data.get("data", []) if isinstance(data, dict) else []:
            if isinstance(item, dict) and item.get("id"):
                names.append(str(item["id"]))
        if not isinstance(data, dict) or not data.get("has_more"):
            return names
        after_id = str(data.get("last_id") or "")
        if not after_id:
            return names


def get_stats_today() -> dict[str, Any]:
    with session_scope() as session:
        stats = session.get(StatsDaily, today_str())
        return stats.to_dict(False) if stats else zero_stats({"date": today_str()})


def get_stats_daily() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [s.to_dict(False) for s in session.scalars(select(StatsDaily).order_by(StatsDaily.date.desc())).all()]


def get_stats_hourly() -> list[dict[str, Any]]:
    today = today_str()
    hour_now = current_hour()
    with session_scope() as session:
        rows = session.scalars(select(StatsHourly).where(StatsHourly.date == today)).all()
        by_hour = {row.hour: row.to_dict(False) for row in rows}
        return [by_hour.get(hour, zero_stats({"hour": hour, "date": today})) for hour in range(hour_now + 1)]


def get_stats_total() -> dict[str, Any]:
    with session_scope() as session:
        stats = ensure_total_stats(session)
        return stats.to_dict(False)


def get_stats_api_keys() -> list[dict[str, Any]]:
    with session_scope() as session:
        return [s.to_dict(False) for s in session.scalars(select(StatsAPIKey).order_by(StatsAPIKey.api_key_id)).all()]


def _price_for_model(session: Session, model_name: str) -> LLMInfo:
    item = session.get(LLMInfo, model_name.lower())
    if item is None:
        item = LLMInfo(name=model_name.lower())
        session.add(item)
        session.flush()
    return item


def record_usage(
    session: Session,
    *,
    api_key_id: int,
    channel_id: int,
    actual_model: str,
    prompt_tokens: int,
    completion_tokens: int,
    wait_time: int,
    success: bool,
) -> float:
    price = _price_for_model(session, actual_model or "unknown")
    input_cost = prompt_tokens * (price.input or 0) / 1_000_000
    output_cost = completion_tokens * (price.output or 0) / 1_000_000
    metrics = {
        "input_token": prompt_tokens,
        "output_token": completion_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "wait_time": wait_time,
        "request_success": 1 if success else 0,
        "request_failed": 0 if success else 1,
    }
    ensure_total_stats(session).add_metrics(metrics)
    daily = session.get(StatsDaily, today_str())
    if daily is None:
        daily = StatsDaily(date=today_str())
        session.add(daily)
    daily.add_metrics(metrics)
    hour_key = current_hour()
    hourly = session.get(StatsHourly, hour_key)
    if hourly is None:
        hourly = StatsHourly(hour=hour_key, date=today_str())
        session.add(hourly)
    elif hourly.date != today_str():
        hourly.date = today_str()
        hourly.input_token = 0
        hourly.output_token = 0
        hourly.input_cost = 0.0
        hourly.output_cost = 0.0
        hourly.wait_time = 0
        hourly.request_success = 0
        hourly.request_failed = 0
    hourly.add_metrics(metrics)
    if api_key_id:
        api_stats = session.get(StatsAPIKey, api_key_id)
        if api_stats is None:
            api_stats = StatsAPIKey(api_key_id=api_key_id)
            session.add(api_stats)
        api_stats.add_metrics(metrics)
    if channel_id:
        channel_stats = session.get(StatsChannel, channel_id)
        if channel_stats is None:
            channel_stats = StatsChannel(channel_id=channel_id)
            session.add(channel_stats)
        channel_stats.add_metrics(metrics)
    model_stats = session.scalar(
        select(StatsModel).where(StatsModel.name == actual_model, StatsModel.channel_id == channel_id).limit(1)
    )
    if model_stats is None:
        model_stats = StatsModel(name=actual_model, channel_id=channel_id)
        session.add(model_stats)
    model_stats.add_metrics(metrics)
    return input_cost + output_cost


def add_relay_log(log_data: dict[str, Any]) -> dict[str, Any]:
    if not log_data.get("id"):
        log_data["id"] = now_ms() * 1000 + random.randint(0, 999)
    with session_scope() as session:
        log = RelayLog(**log_data)
        session.add(log)
        session.flush()
        data = log.to_dict(False)
    for queue in list(relay_subscribers):
        try:
            queue.put_nowait(data)
        except Exception:
            relay_subscribers.discard(queue)
    return data


def list_relay_logs(page: int = 1, page_size: int = 20, start_time: int | None = None, end_time: int | None = None) -> list[dict[str, Any]]:
    page = max(page, 1)
    page_size = 20 if page_size < 1 or page_size > 100 else page_size
    with session_scope() as session:
        stmt = select(RelayLog)
        if start_time is not None and end_time is not None:
            stmt = stmt.where(RelayLog.time >= start_time, RelayLog.time <= end_time)
        rows = session.scalars(stmt.order_by(RelayLog.time.desc()).offset((page - 1) * page_size).limit(page_size)).all()
        return [r.to_dict(False) for r in rows]


def clear_relay_logs() -> None:
    with session_scope() as session:
        session.execute(delete(RelayLog))


def cleanup_old_relay_logs() -> int:
    with session_scope() as session:
        enabled = get_setting(session, "relay_log_keep_enabled", "true").lower() == "true"
        if not enabled:
            return 0
        keep_days = max(get_setting_int(session, "relay_log_keep_period", 7), 1)
        cutoff = int((datetime.now() - timedelta(days=keep_days)).timestamp())
        result = session.execute(delete(RelayLog).where(RelayLog.time < cutoff))
        return int(result.rowcount or 0)


def issue_stream_token() -> str:
    token = generate_stream_token()
    relay_stream_tokens.add(token)
    return token


def consume_stream_token(token: str) -> bool:
    if token in relay_stream_tokens:
        relay_stream_tokens.remove(token)
        return True
    return False


def export_db(include_logs: bool, include_stats: bool) -> dict[str, Any]:
    with session_scope() as session:
        dump: dict[str, Any] = {
            "version": 1,
            "exported_at": iso_now(),
            "include_logs": include_logs,
            "include_stats": include_stats,
            "channels": [x.to_dict(False) for x in session.scalars(select(Channel)).all()],
            "channel_keys": [x.to_dict(False) for x in session.scalars(select(ChannelKey)).all()],
            "groups": [x.to_dict(False) for x in session.scalars(select(Group)).all()],
            "group_items": [x.to_dict(False) for x in session.scalars(select(GroupItem)).all()],
            "llm_infos": [x.to_dict(False) for x in session.scalars(select(LLMInfo)).all()],
            "api_keys": [x.to_dict(False) for x in session.scalars(select(APIKey)).all()],
            "settings": [x.to_dict(False) for x in session.scalars(select(Setting)).all()],
            "stats_total": [],
            "stats_daily": [],
            "stats_hourly": [],
            "stats_model": [],
            "stats_channel": [],
            "stats_api_key": [],
            "relay_logs": [],
        }
        if include_stats:
            dump.update(
                {
                    "stats_total": [x.to_dict(False) for x in session.scalars(select(StatsTotal)).all()],
                    "stats_daily": [x.to_dict(False) for x in session.scalars(select(StatsDaily)).all()],
                    "stats_hourly": [x.to_dict(False) for x in session.scalars(select(StatsHourly)).all()],
                    "stats_model": [x.to_dict(False) for x in session.scalars(select(StatsModel)).all()],
                    "stats_channel": [x.to_dict(False) for x in session.scalars(select(StatsChannel)).all()],
                    "stats_api_key": [x.to_dict(False) for x in session.scalars(select(StatsAPIKey)).all()],
                }
            )
        if include_logs:
            dump["relay_logs"] = [x.to_dict(False) for x in session.scalars(select(RelayLog)).all()]
        return dump


def import_db(dump: dict[str, Any]) -> dict[str, Any]:
    if not dump.get("version") and "data" in dump and isinstance(dump["data"], dict):
        dump = dump["data"]
    rows: dict[str, int] = {}
    model_map = {
        "settings": (Setting, "key"),
        "llm_infos": (LLMInfo, "name"),
        "api_keys": (APIKey, "id"),
        "channels": (Channel, "id"),
        "channel_keys": (ChannelKey, "id"),
        "groups": (Group, "id"),
        "group_items": (GroupItem, "id"),
        "stats_total": (StatsTotal, "id"),
        "stats_daily": (StatsDaily, "date"),
        "stats_hourly": (StatsHourly, "hour"),
        "stats_model": (StatsModel, "id"),
        "stats_channel": (StatsChannel, "channel_id"),
        "stats_api_key": (StatsAPIKey, "api_key_id"),
        "relay_logs": (RelayLog, "id"),
    }
    with session_scope() as session:
        for key, (model, pk) in model_map.items():
            count = 0
            for raw in dump.get(key, []) or []:
                if not isinstance(raw, dict):
                    continue
                data = dict(raw)
                data.pop("keys", None)
                data.pop("items", None)
                data.pop("stats", None)
                existing = session.get(model, data.get(pk)) if data.get(pk) is not None else None
                if existing:
                    for k, v in data.items():
                        if hasattr(existing, k):
                            setattr(existing, k, v)
                else:
                    session.add(model(**data))
                count += 1
            rows[key] = count
        ensure_settings(session)
        ensure_admin_user(session)
        ensure_total_stats(session)
    return {"rows_affected": rows}


async def latest_release_info() -> dict[str, Any]:
    # Python 版是獨立移植版，不再拿 Go upstream release 當「最新版本」，
    # 避免 Web UI 把 `v0.9.28-python.1` 與 upstream `v0.9.28` 誤判為版本不匹配。
    return {
        "tag_name": VERSION,
        "published_at": "",
        "body": "Octopus Python port release",
        "message": "python port latest version",
    }


def update_core() -> str:
    return "update success"
