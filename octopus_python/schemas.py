from __future__ import annotations

import json
from typing import Any, TypeVar

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

T = TypeVar("T")


class ApiErrorMessages:
    INVALID_REQUEST = "Invalid request parameters"
    INVALID_JSON = "Invalid JSON format"
    INVALID_PARAMETER = "Invalid parameter"
    VALIDATION_FAILED = "Input validation failed"
    EXISTS = "Resource already exists"
    NOT_FOUND = "Resource not found"
    INTERNAL = "An unexpected error occurred"
    DATABASE = "Database operation failed"
    AUTH_FAILED = "Authentication failed"


def success(data: Any = None) -> JSONResponse:
    return JSONResponse(status_code=200, content={"code": 200, "message": "success", "data": jsonable(data)})


def error(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": status_code, "message": message})


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class CamelModel(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True, extra="ignore")


class BaseUrl(CamelModel):
    url: str = ""
    delay: int = 0


class CustomHeader(CamelModel):
    header_key: str = ""
    header_value: str = ""


class ChannelKeyIn(CamelModel):
    enabled: bool = True
    channel_key: str = ""
    remark: str = ""


class ChannelKeyUpdateIn(CamelModel):
    id: int
    enabled: bool | None = None
    channel_key: str | None = None
    remark: str | None = None


class UserLoginRequest(CamelModel):
    username: str
    password: str
    expire: int = 0


class ChangePasswordRequest(CamelModel):
    old_password: str
    new_password: str


class ChangeUsernameRequest(CamelModel):
    new_username: str


class APIKeyCreateRequest(CamelModel):
    name: str = ""
    enabled: bool = True
    expire_at: int = 0
    max_cost: float = 0.0
    supported_models: str = ""


class APIKeyUpdateRequest(APIKeyCreateRequest):
    id: int


class ChannelCreateRequest(CamelModel):
    name: str
    type: str = "openai/chat_completions"
    enabled: bool = True
    base_urls: list[BaseUrl] = Field(default_factory=list)
    keys: list[ChannelKeyIn] = Field(default_factory=list)
    model: str = ""
    custom_model: str = ""
    proxy: bool = False
    auto_sync: bool = False
    auto_group: int = 0
    custom_header: list[CustomHeader] = Field(default_factory=list)
    param_override: str | None = ""
    channel_proxy: str | None = ""
    match_regex: str | None = ""


class ChannelUpdateRequest(CamelModel):
    id: int
    name: str | None = None
    type: str | None = None
    enabled: bool | None = None
    base_urls: list[BaseUrl] | None = None
    model: str | None = None
    custom_model: str | None = None
    proxy: bool | None = None
    auto_sync: bool | None = None
    auto_group: int | None = None
    custom_header: list[CustomHeader] | None = None
    param_override: str | None = None
    channel_proxy: str | None = None
    match_regex: str | None = None
    keys_to_add: list[ChannelKeyIn] = Field(default_factory=list)
    keys_to_update: list[ChannelKeyUpdateIn] = Field(default_factory=list)
    keys_to_delete: list[int] = Field(default_factory=list)


class ChannelEnableRequest(CamelModel):
    id: int
    enabled: bool


class FetchModelRequest(CamelModel):
    type: str
    base_urls: list[BaseUrl] = Field(default_factory=list)
    keys: list[ChannelKeyIn] = Field(default_factory=list)
    proxy: bool = False
    channel_proxy: str | None = ""
    match_regex: str | None = ""
    custom_header: list[CustomHeader] = Field(default_factory=list)


class GroupItemRequest(CamelModel):
    channel_id: int
    model_name: str = ""
    priority: int = 0
    weight: int = 1


class GroupItemUpdateRequest(CamelModel):
    id: int
    priority: int = 0
    weight: int = 1


class GroupCreateRequest(CamelModel):
    name: str
    mode: int = 1
    match_regex: str = ""
    first_token_time_out: int = 0
    session_keep_time: int = 0
    items: list[GroupItemRequest] = Field(default_factory=list)


class GroupUpdateRequest(CamelModel):
    id: int
    name: str | None = None
    mode: int | None = None
    match_regex: str | None = None
    first_token_time_out: int | None = None
    session_keep_time: int | None = None
    items_to_add: list[GroupItemRequest] = Field(default_factory=list)
    items_to_update: list[GroupItemUpdateRequest] = Field(default_factory=list)
    items_to_delete: list[int] = Field(default_factory=list)


class LLMInfoRequest(CamelModel):
    name: str
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


class LLMDeleteRequest(CamelModel):
    name: str


class SettingRequest(CamelModel):
    key: str
    value: str


class RelayContext(CamelModel):
    request_type: str
    api_key_id: int
    api_key_name: str
    api_key: str
    supported_models: str = ""


def safe_json_loads(text: str | None, default: Any) -> Any:
    if text in (None, ""):
        return default
    try:
        return json.loads(text)
    except Exception:
        return default


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge_dict(out[key], value)
        else:
            out[key] = value
    return out
