from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.exc import IntegrityError

from . import VERSION
from .config import AppConfig
from .relay import handle_relay
from .schemas import (
    APIKeyCreateRequest,
    APIKeyUpdateRequest,
    ChangePasswordRequest,
    ChangeUsernameRequest,
    ChannelCreateRequest,
    ChannelEnableRequest,
    ChannelUpdateRequest,
    FetchModelRequest,
    GroupCreateRequest,
    GroupUpdateRequest,
    LLMDeleteRequest,
    LLMInfoRequest,
    RelayContext,
    SettingRequest,
    UserLoginRequest,
    split_csv,
    success,
)
from .security import API_KEY_PREFIX, generate_jwt, verify_jwt
from .services import (
    clear_relay_logs,
    consume_stream_token,
    create_api_key,
    create_channel,
    create_group,
    create_llm_info,
    delete_api_key,
    delete_channel,
    delete_group,
    delete_llm_info,
    export_db,
    fetch_models,
    get_api_key_by_key,
    get_api_key_stats,
    get_last_channel_sync_time,
    get_last_model_update_time,
    get_setting,
    get_stats_api_keys,
    get_stats_daily,
    get_stats_hourly,
    get_stats_today,
    get_stats_total,
    get_user,
    import_db,
    issue_stream_token,
    latest_release_info,
    list_api_keys,
    list_channels,
    list_groups,
    list_llm_channels,
    list_llm_infos,
    list_relay_logs,
    list_settings,
    relay_subscribers,
    set_setting,
    sync_channels,
    update_api_key,
    update_channel,
    update_core,
    update_group,
    update_llm_info,
    update_model_prices,
    verify_user,
    change_password as svc_change_password,
    change_username as svc_change_username,
)
from .database import session_scope


def _message(exc: Exception) -> str:
    return str(exc) or exc.__class__.__name__


def api_error(status_code: int, message: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail={"code": status_code, "message": message})


async def admin_auth(authorization: str | None = Header(default=None, alias="Authorization")) -> dict[str, Any]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise api_error(400, "missing token")
    token = authorization.split(" ", 1)[1].strip()
    user = get_user()
    if user is None:
        raise api_error(401, "Authentication failed")
    payload = verify_jwt(token, user.username, user.password)
    if not payload:
        raise api_error(401, "Authentication failed")
    return {"id": user.id, "username": user.username}


async def api_key_auth(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
) -> RelayContext:
    request_type = "anthropic" if x_api_key else "openai"
    key = x_api_key or ""
    if not key and authorization and authorization.lower().startswith("bearer "):
        key = authorization.split(" ", 1)[1].strip()
    if not key or not key.startswith(API_KEY_PREFIX):
        raise api_error(401, "Authentication failed")
    data = get_api_key_by_key(key)
    if not data:
        raise api_error(401, "Authentication failed")
    if not data.get("enabled", True):
        raise api_error(403, "api key disabled")
    expire_at = int(data.get("expire_at") or 0)
    if expire_at > 0 and expire_at < int(time.time()):
        raise api_error(403, "api key expired")
    max_cost = float(data.get("max_cost") or 0)
    if max_cost > 0 and float(data.get("used_cost") or 0) >= max_cost:
        raise api_error(403, "api key cost limit exceeded")
    return RelayContext(
        request_type=request_type,
        api_key_id=int(data["id"]),
        api_key_name=str(data.get("name") or ""),
        api_key=key,
        supported_models=str(data.get("supported_models") or ""),
    )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        if isinstance(exc.detail, dict) and "message" in exc.detail:
            return JSONResponse(status_code=exc.status_code, content=exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"code": exc.status_code, "message": str(exc.detail)})

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        return JSONResponse(status_code=400, content={"code": 400, "message": str(exc)})

    @app.exception_handler(IntegrityError)
    async def integrity_exception_handler(request: Request, exc: IntegrityError):
        return JSONResponse(status_code=409, content={"code": 409, "message": "Resource already exists"})

    @app.exception_handler(LookupError)
    async def lookup_exception_handler(request: Request, exc: LookupError):
        return JSONResponse(status_code=404, content={"code": 404, "message": _message(exc)})

    @app.exception_handler(ValueError)
    async def value_exception_handler(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"code": 400, "message": _message(exc)})

    @app.exception_handler(FileExistsError)
    async def exists_exception_handler(request: Request, exc: FileExistsError):
        return JSONResponse(status_code=409, content={"code": 409, "message": _message(exc)})


def _allowed_origin(origin: str | None) -> bool:
    if not origin:
        return False
    try:
        with session_scope() as session:
            raw = get_setting(session, "cors_allow_origins", "")
    except Exception:
        raw = ""
    if not raw:
        return False
    if raw.strip() == "*":
        return True
    allowed = [x.strip() for x in raw.split(",") if x.strip()]
    if origin in allowed:
        return True
    host = origin.split("://", 1)[-1]
    return host in allowed


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Octopus Python", version=VERSION)
    register_exception_handlers(app)

    # Let the original setting decide CORS at request time. Empty setting denies browser CORS, same as Go middleware.
    @app.middleware("http")
    async def dynamic_cors(request: Request, call_next):
        if request.method == "OPTIONS":
            response = Response(status_code=204)
        else:
            response = await call_next(request)
        origin = request.headers.get("origin")
        if _allowed_origin(origin):
            response.headers["Access-Control-Allow-Origin"] = origin or "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = request.headers.get("access-control-request-headers", "*")
            response.headers["Access-Control-Expose-Headers"] = "Content-Disposition"
        return response

    register_routes(app)
    mount_static(app)
    return app


def register_routes(app: FastAPI) -> None:
    @app.post("/api/v1/user/login")
    async def user_login(req: UserLoginRequest):
        user = verify_user(req.username, req.password)
        if not user:
            raise api_error(401, "Authentication failed")
        token, expire_at = generate_jwt(user.username, user.password, req.expire)
        return success({"token": token, "expire_at": expire_at.isoformat().replace("+00:00", "Z")})

    @app.post("/api/v1/user/change-password")
    async def user_change_password(req: ChangePasswordRequest, _: dict[str, Any] = Depends(admin_auth)):
        svc_change_password(req.old_password, req.new_password)
        return success("password changed successfully")

    @app.post("/api/v1/user/change-username")
    async def user_change_username(req: ChangeUsernameRequest, _: dict[str, Any] = Depends(admin_auth)):
        svc_change_username(req.new_username)
        return success("username changed successfully")

    @app.get("/api/v1/user/status")
    async def user_status(_: dict[str, Any] = Depends(admin_auth)):
        return success("ok")

    @app.post("/api/v1/apikey/create")
    async def apikey_create(req: APIKeyCreateRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(create_api_key(req))

    @app.get("/api/v1/apikey/list")
    async def apikey_list(_: dict[str, Any] = Depends(admin_auth)):
        return success(list_api_keys())

    @app.post("/api/v1/apikey/update")
    async def apikey_update(req: APIKeyUpdateRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(update_api_key(req))

    @app.delete("/api/v1/apikey/delete/{api_key_id}")
    async def apikey_delete(api_key_id: int, _: dict[str, Any] = Depends(admin_auth)):
        delete_api_key(api_key_id)
        return success(None)

    @app.get("/api/v1/apikey/stats")
    async def apikey_stats(ctx: RelayContext = Depends(api_key_auth)):
        return success(get_api_key_stats(ctx.api_key_id))

    @app.get("/api/v1/apikey/login")
    async def apikey_login(ctx: RelayContext = Depends(api_key_auth)):
        return success(None)

    @app.get("/api/v1/channel/list")
    async def channel_list(_: dict[str, Any] = Depends(admin_auth)):
        return success(list_channels())

    @app.post("/api/v1/channel/create")
    async def channel_create(req: ChannelCreateRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(create_channel(req))

    @app.post("/api/v1/channel/update")
    async def channel_update(req: ChannelUpdateRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(update_channel(req))

    @app.post("/api/v1/channel/enable")
    async def channel_enable(req: ChannelEnableRequest, _: dict[str, Any] = Depends(admin_auth)):
        from .services import enable_channel

        enable_channel(req.id, req.enabled)
        return success(None)

    @app.delete("/api/v1/channel/delete/{channel_id}")
    async def channel_delete(channel_id: int, _: dict[str, Any] = Depends(admin_auth)):
        delete_channel(channel_id)
        return success(None)

    @app.post("/api/v1/channel/fetch-model")
    async def channel_fetch_model(req: FetchModelRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(await fetch_models(req))

    @app.post("/api/v1/channel/sync")
    async def channel_sync(_: dict[str, Any] = Depends(admin_auth)):
        await sync_channels()
        return success(None)

    @app.get("/api/v1/channel/last-sync-time")
    async def channel_last_sync(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_last_channel_sync_time())

    @app.get("/api/v1/group/list")
    async def group_list(_: dict[str, Any] = Depends(admin_auth)):
        return success(list_groups())

    @app.post("/api/v1/group/create")
    async def group_create(req: GroupCreateRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(create_group(req))

    @app.post("/api/v1/group/update")
    async def group_update(req: GroupUpdateRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(update_group(req))

    @app.delete("/api/v1/group/delete/{group_id}")
    async def group_delete(group_id: int, _: dict[str, Any] = Depends(admin_auth)):
        delete_group(group_id)
        return success(None)

    @app.get("/api/v1/model/list")
    async def model_list(_: dict[str, Any] = Depends(admin_auth)):
        return success(list_llm_infos())

    @app.post("/api/v1/model/create")
    async def model_create(req: LLMInfoRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(create_llm_info(req))

    @app.get("/api/v1/model/channel")
    async def model_channel(_: dict[str, Any] = Depends(admin_auth)):
        return success(list_llm_channels())

    @app.post("/api/v1/model/update")
    async def model_update(req: LLMInfoRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(update_llm_info(req))

    @app.post("/api/v1/model/delete")
    async def model_delete(req: LLMDeleteRequest, _: dict[str, Any] = Depends(admin_auth)):
        delete_llm_info(req.name)
        return success(None)

    @app.post("/api/v1/model/update-price")
    async def model_update_price(_: dict[str, Any] = Depends(admin_auth)):
        await update_model_prices()
        return success(None)

    @app.get("/api/v1/model/last-update-time")
    async def model_last_update(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_last_model_update_time())

    @app.get("/v1/models")
    async def relay_models(ctx: RelayContext = Depends(api_key_auth)):
        names = [g["name"] for g in list_groups()]
        supported = split_csv(ctx.supported_models)
        if supported:
            names = [n for n in names if n in supported]
        if ctx.request_type == "anthropic":
            data = [
                {"id": n, "created_at": "2024-01-01T00:00:00Z", "display_name": n, "type": "model"}
                for n in names
            ]
            return {"data": data, "has_more": False, "first_id": data[0]["id"] if data else "", "last_id": data[-1]["id"] if data else ""}
        return {"success": True, "data": [{"id": n, "object": "model", "created": 1763395200, "owned_by": "octopus"} for n in names], "object": "list"}

    @app.get("/api/v1/stats/today")
    async def stats_today(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_stats_today())

    @app.get("/api/v1/stats/daily")
    async def stats_daily(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_stats_daily())

    @app.get("/api/v1/stats/hourly")
    async def stats_hourly(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_stats_hourly())

    @app.get("/api/v1/stats/total")
    async def stats_total(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_stats_total())

    @app.get("/api/v1/stats/apikey")
    async def stats_apikey(_: dict[str, Any] = Depends(admin_auth)):
        return success(get_stats_api_keys())

    @app.get("/api/v1/setting/list")
    async def setting_list(_: dict[str, Any] = Depends(admin_auth)):
        return success(list_settings())

    @app.post("/api/v1/setting/set")
    async def setting_set(req: SettingRequest, _: dict[str, Any] = Depends(admin_auth)):
        return success(set_setting(req))

    @app.get("/api/v1/setting/export")
    async def setting_export(
        include_logs: bool = False,
        include_stats: bool = False,
        _: dict[str, Any] = Depends(admin_auth),
    ):
        dump = export_db(include_logs, include_stats)
        content = json.dumps(dump, ensure_ascii=False, indent=2)
        ts = time.strftime("%Y%m%d%H%M%S")
        return Response(
            content=content,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="octopus-export-{ts}.json"'},
        )

    @app.post("/api/v1/setting/import")
    async def setting_import(request: Request, file: UploadFile | None = File(default=None), _: dict[str, Any] = Depends(admin_auth)):
        if file is not None:
            body = await file.read()
        else:
            body = await request.body()
        try:
            dump = json.loads(body.decode("utf-8"))
        except Exception as exc:
            raise api_error(400, str(exc))
        return success(import_db(dump))

    @app.get("/api/v1/log/list")
    async def log_list(
        page: int = 1,
        page_size: int = 20,
        start_time: int | None = None,
        end_time: int | None = None,
        _: dict[str, Any] = Depends(admin_auth),
    ):
        return success(list_relay_logs(page, page_size, start_time, end_time))

    @app.delete("/api/v1/log/clear")
    async def log_clear(_: dict[str, Any] = Depends(admin_auth)):
        clear_relay_logs()
        return success(None)

    @app.get("/api/v1/log/stream-token")
    async def log_stream_token(_: dict[str, Any] = Depends(admin_auth)):
        return success({"token": issue_stream_token()})

    @app.get("/api/v1/log/stream")
    async def log_stream(token: str):
        if not consume_stream_token(token):
            raise api_error(401, "invalid stream token")
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        relay_subscribers.add(queue)

        async def events() -> AsyncGenerator[str, None]:
            try:
                while True:
                    item = await queue.get()
                    yield "data: " + json.dumps(item, ensure_ascii=False) + "\n\n"
            finally:
                relay_subscribers.discard(queue)

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/v1/update")
    async def update_latest(_: dict[str, Any] = Depends(admin_auth)):
        return success(await latest_release_info())

    @app.get("/api/v1/update/now-version")
    async def update_now_version(_: dict[str, Any] = Depends(admin_auth)):
        return success(VERSION)

    @app.post("/api/v1/update")
    async def update_now(_: dict[str, Any] = Depends(admin_auth)):
        return success(update_core())

    @app.post("/v1/chat/completions")
    async def relay_chat(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "chat", ctx)

    @app.post("/v1/responses")
    async def relay_responses(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "responses", ctx)

    @app.post("/v1/messages")
    async def relay_messages(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "messages", ctx)

    @app.post("/v1/embeddings")
    async def relay_embeddings(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "embeddings", ctx)

    @app.post("/v1/images/generations")
    async def relay_images_generations(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "images_generations", ctx)

    @app.post("/v1/images/edits")
    async def relay_images_edits(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "images_edits", ctx)

    @app.post("/v1/images/variations")
    async def relay_images_variations(request: Request, ctx: RelayContext = Depends(api_key_auth)):
        return await handle_relay(request, "images_variations", ctx)


def mount_static(app: FastAPI) -> None:
    candidates = [
        Path.cwd() / "static" / "out",
        Path(__file__).resolve().parent.parent / "static" / "out",
    ]
    static_dir = next((path for path in candidates if path.exists()), None)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
