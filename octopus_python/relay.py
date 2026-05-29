from __future__ import annotations

import json
import random
import time
from typing import Any

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .database import Channel, Group, GroupItem, session_scope
from .schemas import RelayContext, merge_dict, safe_json_loads, split_csv
from .services import (
    SENSITIVE_HEADERS,
    add_relay_log,
    choose_base_url,
    choose_channel_key,
    circuit_breakers,
    get_setting_int,
    record_usage,
    round_robin_counters,
    sticky_sessions,
)


class RelayError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


def extract_usage(data: Any) -> tuple[int, int]:
    if not isinstance(data, dict):
        return 0, 0
    usage = data.get("usage") or {}
    if not isinstance(usage, dict):
        return 0, 0
    prompt = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
    completion = usage.get("completion_tokens") or usage.get("output_tokens") or 0
    try:
        return int(prompt), int(completion)
    except Exception:
        return 0, 0


def truncate_json(value: Any, max_len: int = 12000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False)
    except Exception:
        text = str(value)
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def build_upstream_url(channel_type: str, base_url: str, endpoint: str, model_name: str = "") -> str:
    base = base_url.rstrip("/")
    if channel_type == "gemini/contents":
        if endpoint in {"chat", "responses"}:
            return f"{base}/models/{model_name}:generateContent"
        if endpoint == "embeddings":
            return f"{base}/models/{model_name}:embedContent"
    if channel_type == "anthropic/messages":
        return base + "/messages"
    if channel_type == "openai/responses":
        return base + "/responses"
    if channel_type == "openai/embeddings":
        return base + "/embeddings"
    if channel_type.startswith("openai/images"):
        if endpoint == "images_edits":
            return base + "/images/edits"
        if endpoint == "images_variations":
            return base + "/images/variations"
        return base + "/images/generations"
    if endpoint == "images_generations":
        return base + "/images/generations"
    if endpoint == "images_edits":
        return base + "/images/edits"
    if endpoint == "images_variations":
        return base + "/images/variations"
    return base + "/chat/completions"


def openai_messages_to_gemini(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for msg in messages or []:
        role = msg.get("role", "user")
        if role == "assistant":
            role = "model"
        elif role == "system":
            role = "user"
        content = msg.get("content", "")
        parts: list[dict[str, Any]] = []
        if isinstance(content, str):
            parts.append({"text": content})
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        parts.append({"text": part.get("text", "")})
                    elif "text" in part:
                        parts.append({"text": part.get("text", "")})
        contents.append({"role": role, "parts": parts or [{"text": ""}]})
    return contents


def to_gemini_payload(body: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"contents": openai_messages_to_gemini(body.get("messages") or [])}
    generation: dict[str, Any] = {}
    for src, dst in [("temperature", "temperature"), ("top_p", "topP"), ("max_tokens", "maxOutputTokens")]:
        if src in body:
            generation[dst] = body[src]
    if generation:
        payload["generationConfig"] = generation
    return payload


def gemini_to_openai(data: dict[str, Any], model: str, stream: bool = False) -> dict[str, Any]:
    text = ""
    candidates = data.get("candidates") or []
    if candidates:
        parts = (((candidates[0] or {}).get("content") or {}).get("parts") or [])
        text = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
    usage = data.get("usageMetadata") or {}
    return {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }


def openai_to_anthropic_payload(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages") or []
    system_parts: list[str] = []
    ant_messages: list[dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content if isinstance(content, str) else json.dumps(content, ensure_ascii=False))
        elif role in {"user", "assistant"}:
            ant_messages.append({"role": role, "content": content})
    payload: dict[str, Any] = {
        "model": body.get("model"),
        "messages": ant_messages,
        "max_tokens": body.get("max_tokens") or body.get("max_completion_tokens") or 4096,
    }
    if system_parts:
        payload["system"] = "\n".join(system_parts)
    for key in ["temperature", "top_p", "stream", "stop"]:
        if key in body:
            payload[key] = body[key]
    return payload


def anthropic_to_openai(data: dict[str, Any], model: str) -> dict[str, Any]:
    text = ""
    for part in data.get("content") or []:
        if isinstance(part, dict) and part.get("type") == "text":
            text += part.get("text", "")
    usage = data.get("usage") or {}
    return {
        "id": data.get("id", f"chatcmpl-{int(time.time())}"),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": (usage.get("input_tokens", 0) or 0) + (usage.get("output_tokens", 0) or 0),
        },
    }


def prepare_body(channel_type: str, endpoint: str, body: dict[str, Any], actual_model: str) -> tuple[dict[str, Any], bool]:
    out = dict(body)
    out["model"] = actual_model
    converted = False
    if channel_type == "gemini/contents" and endpoint in {"chat", "responses"}:
        return to_gemini_payload(out), True
    if channel_type == "gemini/contents" and endpoint == "embeddings":
        content = body.get("input") or body.get("text") or ""
        if isinstance(content, list):
            content = "\n".join(str(item) for item in content)
        return {"model": f"models/{actual_model}", "content": {"parts": [{"text": str(content)}]}}, True
    if channel_type == "anthropic/messages" and endpoint in {"chat", "responses"}:
        return openai_to_anthropic_payload(out), True
    return out, converted


def prepare_headers(channel_type: str, key: str, channel_headers: list[dict[str, Any]], original: Request) -> dict[str, str]:
    headers: dict[str, str] = {"User-Agent": "octopus-python"}
    if channel_type == "anthropic/messages":
        headers["x-api-key"] = key
        headers["anthropic-version"] = original.headers.get("anthropic-version", "2023-06-01")
    elif channel_type == "gemini/contents":
        headers["x-goog-api-key"] = key
    else:
        headers["Authorization"] = f"Bearer {key}"
    headers["Content-Type"] = original.headers.get("content-type", "application/json")
    for item in channel_headers or []:
        hk = item.get("header_key") if isinstance(item, dict) else None
        hv = item.get("header_value") if isinstance(item, dict) else None
        if hk and hk.lower() not in SENSITIVE_HEADERS:
            headers[str(hk)] = str(hv or "")
    return headers


def is_circuit_open(session, channel_id: int, key_id: int, model: str) -> tuple[bool, str]:
    ckey = f"{channel_id}:{key_id}:{model}"
    state = circuit_breakers.get(ckey)
    if not state:
        return False, ""
    if state.get("state") == "open" and time.time() < state.get("until", 0):
        return True, "circuit breaker open"
    return False, ""


def mark_circuit(session, channel_id: int, key_id: int, model: str, success: bool) -> None:
    ckey = f"{channel_id}:{key_id}:{model}"
    if success:
        circuit_breakers.pop(ckey, None)
        return
    threshold = get_setting_int(session, "circuit_breaker_threshold", 5)
    cooldown = get_setting_int(session, "circuit_breaker_cooldown", 60)
    max_cooldown = get_setting_int(session, "circuit_breaker_max_cooldown", 600)
    state = circuit_breakers.setdefault(ckey, {"failures": 0, "state": "closed", "until": 0, "trips": 0})
    state["failures"] = state.get("failures", 0) + 1
    if state["failures"] >= threshold:
        state["trips"] = state.get("trips", 0) + 1
        state["state"] = "open"
        state["until"] = time.time() + min(max_cooldown, cooldown * (2 ** (state["trips"] - 1)))


def channel_supports_endpoint(channel_type: str, endpoint: str) -> bool:
    if endpoint == "embeddings":
        return channel_type in {
            "openai/chat_completions",
            "openai/responses",
            "openai/embeddings",
            "gemini/contents",
            "doubao",
        }
    if endpoint.startswith("images_"):
        return channel_type in {
            "openai/chat_completions",
            "openai/responses",
            "openai/images_generations",
            "openai/images_edits",
            "openai/images_variations",
            "gemini/contents",
            "doubao",
        }
    return channel_type in {
        "openai/chat_completions",
        "openai/responses",
        "anthropic/messages",
        "gemini/contents",
        "doubao",
    }


def ordered_group_items(group: Group, api_key_id: int, request_model: str) -> list[GroupItem]:
    items = [item for item in group.items if item.channel and item.channel.enabled]
    if group.mode == 2:
        random.shuffle(items)
    elif group.mode == 3:
        items.sort(key=lambda x: x.priority)
    elif group.mode == 4:
        total = sum(max(item.weight, 1) for item in items)
        if total > 0:
            pick = random.randint(1, total)
            running = 0
            chosen_index = 0
            for idx, item in enumerate(items):
                running += max(item.weight, 1)
                if pick <= running:
                    chosen_index = idx
                    break
            items = items[chosen_index:] + items[:chosen_index]
    else:
        if items:
            idx = round_robin_counters[group.id] % len(items)
            round_robin_counters[group.id] += 1
            items = items[idx:] + items[:idx]
    sticky_key = f"{api_key_id}:{request_model}"
    sticky = sticky_sessions.get(sticky_key)
    if sticky and group.session_keep_time > 0 and time.time() - sticky[2] <= group.session_keep_time:
        sticky_channel_id, sticky_key_id, _ = sticky
        items.sort(key=lambda x: 0 if x.channel_id == sticky_channel_id else 1)
    return items


def _model_supported(supported_models: str, model: str) -> bool:
    models = split_csv(supported_models)
    return not models or model in models


async def handle_relay(request: Request, endpoint: str, ctx: RelayContext) -> Response:
    try:
        if endpoint.startswith("images_"):
            body = await parse_any_body(request)
        else:
            body = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid JSON format"}})
    if not isinstance(body, dict):
        return JSONResponse(status_code=400, content={"error": {"message": "Invalid request parameters"}})
    request_model = str(body.get("model") or "")
    if not request_model:
        return JSONResponse(status_code=400, content={"error": {"message": "model is required"}})
    if not _model_supported(ctx.supported_models, request_model):
        return JSONResponse(status_code=400, content={"error": {"message": "model not supported"}})

    start = time.time()
    attempts: list[dict[str, Any]] = []
    last_error = ""

    with session_scope() as session:
        group = session.scalar(
            select(Group)
            .options(selectinload(Group.items).selectinload(GroupItem.channel).selectinload(Channel.keys))
            .where(Group.name == request_model)
            .limit(1)
        )
        if group is None:
            return JSONResponse(status_code=404, content={"error": {"message": "model not found"}})
        items = ordered_group_items(group, ctx.api_key_id, request_model)
        if not items:
            return JSONResponse(status_code=503, content={"error": {"message": "no available channel"}})

        for attempt_num, item in enumerate(items, start=1):
            channel = item.channel
            actual_model = item.model_name or request_model
            if not channel_supports_endpoint(channel.type, endpoint):
                attempts.append(
                    {
                        "channel_id": channel.id,
                        "channel_key_id": 0,
                        "channel_name": channel.name,
                        "model_name": actual_model,
                        "attempt_num": attempt_num,
                        "status": "skipped",
                        "duration": 0,
                        "sticky": False,
                        "msg": f"channel type {channel.type} does not support {endpoint}",
                    }
                )
                continue
            channel_key = choose_channel_key(channel)
            attempt = {
                "channel_id": channel.id,
                "channel_key_id": channel_key.id if channel_key else 0,
                "channel_name": channel.name,
                "model_name": actual_model,
                "attempt_num": attempt_num,
                "status": "failed",
                "duration": 0,
                "sticky": False,
                "msg": "",
            }
            if not channel_key:
                attempt.update({"status": "skipped", "msg": "no available channel key"})
                attempts.append(attempt)
                continue
            open_, msg = is_circuit_open(session, channel.id, channel_key.id, actual_model)
            if open_:
                attempt.update({"status": "circuit_break", "msg": msg})
                attempts.append(attempt)
                continue
            base = choose_base_url(channel)
            if not base:
                attempt.update({"status": "skipped", "msg": "no base url"})
                attempts.append(attempt)
                continue
            upstream_body, converted = prepare_body(channel.type, endpoint, body, actual_model)
            override = safe_json_loads(channel.param_override, {})
            if isinstance(override, dict):
                upstream_body = merge_dict(upstream_body, override)
            url = build_upstream_url(channel.type, base, endpoint, actual_model)
            headers = prepare_headers(channel.type, channel_key.channel_key, channel.custom_header or [], request)
            try:
                duration_start = time.time()
                result, raw_bytes, status_code, media_type = await forward_request(
                    request=request,
                    url=url,
                    headers=headers,
                    body=upstream_body,
                    channel_type=channel.type,
                    endpoint=endpoint,
                    actual_model=actual_model,
                )
                duration = int((time.time() - duration_start) * 1000)
                attempt["duration"] = duration
                success = 200 <= status_code < 400
                attempt["status"] = "success" if success else "failed"
                if not success:
                    attempt["msg"] = raw_bytes.decode("utf-8", errors="ignore")[:500]
                    last_error = attempt["msg"]
                    channel_key.status_code = status_code
                    channel_key.last_use_time_stamp = int(time.time())
                    mark_circuit(session, channel.id, channel_key.id, actual_model, False)
                    attempts.append(attempt)
                    continue
                prompt, completion = extract_usage(result)
                use_time = int((time.time() - start) * 1000)
                cost = record_usage(
                    session,
                    api_key_id=ctx.api_key_id,
                    channel_id=channel.id,
                    actual_model=actual_model,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    wait_time=use_time,
                    success=True,
                )
                channel_key.status_code = status_code
                channel_key.last_use_time_stamp = int(time.time())
                channel_key.total_cost = (channel_key.total_cost or 0) + cost
                mark_circuit(session, channel.id, channel_key.id, actual_model, True)
                sticky_sessions[f"{ctx.api_key_id}:{request_model}"] = (channel.id, channel_key.id, time.time())
                attempts.append(attempt)
                log_data = {
                    "id": 0,
                    "time": int(time.time()),
                    "request_model_name": request_model,
                    "request_api_key_name": ctx.api_key_name,
                    "channel": channel.id,
                    "channel_name": channel.name,
                    "actual_model_name": actual_model,
                    "input_tokens": prompt,
                    "output_tokens": completion,
                    "ftut": 0,
                    "use_time": use_time,
                    "cost": cost,
                    "request_content": truncate_json(body),
                    "response_content": truncate_json(result if result is not None else raw_bytes.decode("utf-8", "ignore")),
                    "error": "",
                    "attempts": attempts,
                    "total_attempts": len(attempts),
                }
                session.flush()
                # Add log in a nested independent transaction after current changes commit.
                session.expunge_all()
                add_relay_log(log_data)
                if endpoint.startswith("images_") and result is None:
                    return Response(content=raw_bytes, status_code=status_code, media_type=media_type)
                return build_downstream_response(result, raw_bytes, status_code, media_type, request, ctx)
            except Exception as exc:
                attempt["duration"] = int((time.time() - start) * 1000)
                attempt["status"] = "failed"
                attempt["msg"] = str(exc)
                last_error = str(exc)
                mark_circuit(session, channel.id, channel_key.id, actual_model, False)
                attempts.append(attempt)
                continue

        use_time = int((time.time() - start) * 1000)
        record_usage(
            session,
            api_key_id=ctx.api_key_id,
            channel_id=0,
            actual_model=request_model,
            prompt_tokens=0,
            completion_tokens=0,
            wait_time=use_time,
            success=False,
        )
        session.flush()
    add_relay_log(
        {
            "id": 0,
            "time": int(time.time()),
            "request_model_name": request_model,
            "request_api_key_name": ctx.api_key_name,
            "channel": 0,
            "channel_name": "",
            "actual_model_name": request_model,
            "input_tokens": 0,
            "output_tokens": 0,
            "ftut": 0,
            "use_time": int((time.time() - start) * 1000),
            "cost": 0.0,
            "request_content": truncate_json(body),
            "response_content": "",
            "error": last_error or "all channel attempts failed",
            "attempts": attempts,
            "total_attempts": len(attempts),
        }
    )
    return JSONResponse(status_code=502, content={"error": {"message": last_error or "all channel attempts failed"}})


async def parse_any_body(request: Request) -> dict[str, Any]:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        return await request.json()
    if "multipart/form-data" in ctype or "application/x-www-form-urlencoded" in ctype:
        form = await request.form()
        out: dict[str, Any] = {}
        for key, value in form.multi_items():
            if hasattr(value, "filename"):
                continue
            out[key] = value
        return out
    try:
        return await request.json()
    except Exception:
        return {}


async def forward_request(
    *,
    request: Request,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    channel_type: str,
    endpoint: str,
    actual_model: str,
) -> tuple[Any, bytes, int, str]:
    stream = bool(body.get("stream")) and not endpoint.startswith("images_")
    timeout = httpx.Timeout(300.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        if stream:
            # Non-buffered upstream streaming. Logs cannot know final usage here, but response is preserved.
            res = await client.post(url, headers=headers, json=body)
        else:
            res = await client.post(url, headers=headers, json=body)
    raw = res.content
    media_type = res.headers.get("content-type", "application/json")
    parsed: Any = None
    if "application/json" in media_type:
        try:
            parsed = res.json()
        except Exception:
            parsed = None
    if parsed is not None and 200 <= res.status_code < 400:
        if channel_type == "gemini/contents" and endpoint in {"chat", "responses"}:
            parsed = gemini_to_openai(parsed, actual_model)
            raw = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            media_type = "application/json"
        elif channel_type == "anthropic/messages" and endpoint in {"chat", "responses"} and request.url.path.endswith("/chat/completions"):
            parsed = anthropic_to_openai(parsed, actual_model)
            raw = json.dumps(parsed, ensure_ascii=False).encode("utf-8")
            media_type = "application/json"
    return parsed, raw, res.status_code, media_type


def build_downstream_response(
    result: Any, raw: bytes, status_code: int, media_type: str, request: Request, ctx: RelayContext
) -> Response:
    # If caller used x-api-key for Anthropic, return provider/native payload. Otherwise return OpenAI-compatible payload.
    if result is not None and "application/json" in media_type:
        return JSONResponse(status_code=status_code, content=result)
    return Response(content=raw, status_code=status_code, media_type=media_type)
